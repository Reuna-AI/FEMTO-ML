// esp-yolov5-print.ino
// Converted for the new NHWC int8 compiler (run_model + union arena). LCD removed.
//
// Build the model with:  python build_model.py yolov5n.onnx gencode/ --target esp32s3 --simd
// Then add gencode/model.c and gencode/model.h to this sketch folder (or include them).
//
// The new compiler does NOT take forward_pass(in,out). Instead:
//   - activations_init()  allocates the PSRAM tensor arena  (call once)
//   - you fill the INPUT tensor buffer  (tensor_images, the graph input)
//   - run_model()         executes the whole graph in-place over the arena
//   - you read the OUTPUT tensor buffer (the graph output)  -> parse detections
//
// IMPORTANT — set these two names to match YOUR generated model.h:
//   MODEL_INPUT   : the input tensor macro (grep '#define tensor_' activations.h ; it's the graph input)
//   MODEL_OUTPUT  : the output tensor macro (the final graph output feeding the head)
// and the input dtype: the new pipeline is int8 NHWC, so input is int8 [1,224,224,3].

#include "model.h"          // <-- generated: pulls in weights.h + activations.h, declares run_model()
#include "images.h"

// ---- MUST MATCH your generated names / shapes ----
#define MODEL_OUTPUT  tensor_output0         // <-- rename to your real output macro

// input is FLOAT NCHW [1][3][224][224] — the model quantizes internally
#define MODEL_INPUT   tensor_images        // float (*)[3][224][224], macro (*tensor_images_p)
#define IN_H 224
#define IN_W 224

#define IN_C 3
// input quantization (from the first QuantizeLinear in your model): y = round(x/scale)+zp
#define IN_SCALE   0.00392156862745098f      // 1/255
#define IN_ZP      (-128)

// arena init/free are DEFINED in the generated code; just declare here.
int  activations_init(void);
void activations_free(void);
void run_model(void);

// ---------------- geometry / detection config ----------------
#define I_HEIGHT 250
#define I_WIDTH  250
#define DST_WIDTH  224
#define DST_HEIGHT 224
#define IOU_THRESHOLD 0.45f
#define original_width  420
#define original_height 320
#define NUM_BOXES 3087
#define NUM_CLASSES 80
#define CONFIDENCE_THRESHOLD 0.5f

#define IMG_H 224
#define IMG_W 224


typedef struct { float x,y,w,h; float confidence; float class_scores; int class_id; } Detection;

float compute_iou(Detection a, Detection b){
  int x_left=fmax(a.x,b.x), y_top=fmax(a.y,b.y);
  int x_right=fmin(a.x+a.w,b.x+b.w), y_bottom=fmin(a.y+a.h,b.y+b.h);
  if(x_right<x_left||y_bottom<y_top) return 0.0f;
  int inter=(x_right-x_left)*(y_bottom-y_top);
  int ua=a.w*a.h, ub=b.w*b.h;
  return (float)inter/(float)(ua+ub-inter);
}

void non_maximum_suppression(Detection d[], int *n, float thr){
  for(int i=0;i<*n;i++){ if(d[i].confidence<=0) continue;
    for(int j=i+1;j<*n;j++){ if(d[j].confidence<=0) continue;
      if(compute_iou(d[i],d[j])>thr) d[j].confidence=0; } }
  int m=0; for(int i=0;i<*n;i++) if(d[i].confidence>0) d[m++]=d[i]; *n=m;
}

void parse_yolo_output(float output[NUM_BOXES][85], Detection det[], int *dc){
  *dc=0;
  float sx=(float)original_width/DST_WIDTH, sy=(float)original_height/DST_HEIGHT;
  for(int i=0;i<NUM_BOXES;i++){
    float conf=output[i][4]; if(conf<CONFIDENCE_THRESHOLD) continue;
    float best=-INFINITY; int cid=-1;
    for(int j=5;j<85;j++) if(output[i][j]>best){ best=output[i][j]; cid=j-5; }
    float comb=best*conf; if(comb<0.4f) continue;
    float cx=output[i][0],cy=output[i][1],w=output[i][2],h=output[i][3];
    int x0=(int)((cx-w/2)*sx), y0=(int)((cy-h/2)*sy);
    int x1=(int)((cx+w/2)*sx), y1=(int)((cy+h/2)*sy);
    x0=fmax(0,fmin(x0,original_width-1)); y0=fmax(0,fmin(y0,original_height-1));
    x1=fmax(0,fmin(x1,original_width-1)); y1=fmax(0,fmin(y1,original_height-1));
    det[*dc].x=x0; det[*dc].y=y0; det[*dc].w=x1-x0; det[*dc].h=y1-y0;
    det[*dc].class_scores=comb; det[*dc].class_id=cid; det[*dc].confidence=conf;
    (*dc)++;
  }
  non_maximum_suppression(det,dc,IOU_THRESHOLD);
}

// ---- resize RGB565 source -> 224x224 (nearest) ----
void resizeImage(const uint16_t *src, uint16_t *dst){
  for(int y=0;y<DST_HEIGHT;y++) for(int x=0;x<DST_WIDTH;x++){
    int sxp=(x*I_WIDTH)/DST_WIDTH, syp=(y*I_HEIGHT)/DST_HEIGHT;
    dst[y*DST_WIDTH+x]=src[syp*I_WIDTH+sxp];
  }
}

// ---- RGB565 -> quantized int8 NHWC, written straight into the model input tensor ----
// NHWC layout: index = ((h*IN_W)+w)*IN_C + c. Quantize: q = round(norm/scale)+zp, clamp int8.
static inline int8_t quant(float norm){
  int r=(int)lroundf(norm/IN_SCALE)+IN_ZP;
  if(r<-128)r=-128; if(r>127)r=127; return (int8_t)r;
}

// ---- optional: dump the EXACT input you fed, so you can diff against ORT ----
// Call AFTER fill_input and BEFORE run_model(). Capture between the markers,
// then in Python: reshape to (1,3,224,224) and np.save for compare_ort --input-npy.
// ---- dump the exact input you fed, for compare_ort.py --input-npy ----
void dump_input_nchw() {
  Serial.println("INPUT_DUMP_BEGIN");
  for (int c = 0; c < 3; ++c)
    for (int h = 0; h < 5; ++h)
      for (int w = 0; w < 5; ++w)
        Serial.printf("%.6f\n", MODEL_INPUT[c][h][w]);
  Serial.println("INPUT_DUMP_END");
  Serial.flush();
}

// Fill from a resized RGB565 buffer (img[h][w]), NCHW, normalized /255.
void fill_input_from_rgb565(const uint16_t img[DST_HEIGHT][DST_WIDTH]) {
  for (int h = 0; h < 224; ++h) {
    for (int w = 0; w < 224; ++w) {
      uint16_t px = img[h][w];
      // RGB565 -> 5/6/5 fields
      uint8_t r5 = (px >> 11) & 0x1F;
      uint8_t g6 = (px >> 5)  & 0x3F;
      uint8_t b5 =  px        & 0x1F;
      // expand to 8-bit (RGB565 -> RGB888)
      uint8_t r8 = (r5 << 3) | (r5 >> 2);
      uint8_t g8 = (g6 << 2) | (g6 >> 4);
      uint8_t b8 = (b5 << 3) | (b5 >> 2);
      // normalize /255, write NCHW [c][h][w]
      MODEL_INPUT[0][h][w] = (float)r8 / 255.0f;   // channel 0 = R
      MODEL_INPUT[1][h][w] = (float)g8 / 255.0f;   // channel 1 = G
      MODEL_INPUT[2][h][w] = (float)b8 / 255.0f;   // channel 2 = B
    }
  }
}


void fill_input_rgb888(const uint32_t* image) {
  for (int h = 0; h < IMG_H; ++h) {
    for (int w = 0; w < IMG_W; ++w) {
      uint32_t px = image[h * IMG_W + w];      // one packed pixel
      uint8_t r = (px >> 16) & 0xFF;            // 0x00RRGGBB -> R
      uint8_t g = (px >> 8)  & 0xFF;            //           -> G
      uint8_t b =  px        & 0xFF;            //           -> B
      MODEL_INPUT[0][h][w] = (float)r / 255.0f;
      MODEL_INPUT[1][h][w] = (float)g / 255.0f;
      MODEL_INPUT[2][h][w] = (float)b / 255.0f;
    }
  }
}
void setup(){
  Serial.begin(115200);
  delay(1500);

  if(!psramInit()){ Serial.println("PSRAM init failed!"); return; }
  Serial.printf("Free PSRAM: %u B\n", (unsigned)ESP.getFreePsram());

  // allocate the model's tensor arena in PSRAM
  if(activations_init()){ Serial.println("activations_init failed (PSRAM arena)"); return; }

  // ---- prepare input: resize source image to 224x224, quantize into MODEL_INPUT ----
  uint16_t *resized=(uint16_t*)ps_malloc(DST_WIDTH*DST_HEIGHT*sizeof(uint16_t));
  resizeImage(*picture_1, resized);
  uint16_t (*img2d)[DST_WIDTH]=(uint16_t(*)[DST_WIDTH])resized;

  //fill_input_from_rgb565(img2d);

  fill_input_rgb888(image);
  dump_input_nchw();
  free(resized);

  // ---- run inference ----
  Serial.println("running model...");
  unsigned long t0=micros();
  run_model();
  unsigned long t1=micros();
  Serial.printf("Execution time: %lu us  (%.2f ms)\n", t1-t0, (t1-t0)/1000.0);

  // ---- read output ----
  // NOTE: the generated output tensor is int8 and its shape/layout depends on your head.
  // parse_yolo_output expects float[3087][85]. If your model head outputs int8, you must
  // dequantize it into a float[3087][85] buffer first (scale/zp from the output tensor).
  // Below assumes MODEL_OUTPUT is already float[3087][85]-shaped; adjust to your head.
  float (*output)[85] = (float(*)[85]) MODEL_OUTPUT;   // <-- adjust cast to your real output

  Detection *det=(Detection*)ps_malloc(sizeof(Detection)*600);
  int dc=0;
  parse_yolo_output((float(*)[85])output, det, &dc);

  Serial.printf("detections: %d\n", dc);
  for(int i=0;i<dc;i++){
    Serial.printf("[class %d  x=%.0f y=%.0f w=%.0f h=%.0f conf=%.3f score=%.3f]\n",
      det[i].class_id, det[i].x, det[i].y, det[i].w, det[i].h,
      det[i].confidence, det[i].class_scores);
  }
  free(det);

  activations_free();   // optional: free arena if you won't run again
  Serial.println("done.");
}

void loop(){ delay(1000); }
