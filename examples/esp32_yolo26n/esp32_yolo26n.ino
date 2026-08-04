// sketch_jul30b.ino
// YOLO26-style end2end model on ESP32-S3 (NHWC int8 compiler, run_model + union arena).
//
// Build:  python synapedge_reuna.py model.onnx gencode/ --target esp32s3 --simd
// Add gencode/model.c(.cpp), activations.c(.cpp), weights_*.c(.cpp) to this folder.
//
// Flow:
//   activations_init()  -> allocate PSRAM tensor arena (once)
//   fill_input_rgb888() -> write the graph input (float NCHW [1][3][224][224])
//   run_model()         -> executes the whole graph (incl. attention + NMS)
//   read tensor_output0 -> [1][300][6] final detections, NMS already done
//
// Output format: output0 = float[300][6], each row = [x1, y1, x2, y2, score, class_id]
//   boxes are xyxy corners in MODEL INPUT resolution (224x224).

#include "model.h"          // generated: pulls in weights.h + activations.h, declares run_model()
#include "images.h"

// ---- MUST MATCH your generated names / shapes ----
#define MODEL_INPUT   tensor_images        // float (*)[3][224][224], macro (*tensor_images_p)
#define MODEL_OUTPUT  tensor_output0       // float (*)[300][6],      macro (*tensor_output0_p)

#define IN_H 256
#define IN_W 256
#define IN_C 3
#define IMG_H 256
#define IMG_W 256



// ---- detection output config ----
#define NUM_DET       300
#define DET_STRIDE    6           // [x1,y1,x2,y2,score,class]
#define SCORE_THRESH  0.25f       // filter low-score / padding slots

// arena init/free + run are DEFINED in the generated code; declare here.
int  activations_init(void);
void activations_free(void);
void run_model(void);

// ---------------- input preprocessing ----------------
// Fill from a 24-bit packed image[H*W] (uint32 0x00RRGGBB), normalize /255, NCHW.
void fill_input_rgb888(const uint32_t* image) {
  for (int h = 0; h < IMG_H; ++h) {
    for (int w = 0; w < IMG_W; ++w) {
      uint32_t px = image[h * IMG_W + w];
      uint8_t r = (px >> 16) & 0xFF;
      uint8_t g = (px >> 8)  & 0xFF;
      uint8_t b =  px        & 0xFF;
      MODEL_INPUT[0][h][w] = (float)r / 255.0f;   // R
      MODEL_INPUT[1][h][w] = (float)g / 255.0f;   // G
      MODEL_INPUT[2][h][w] = (float)b / 255.0f;   // B
    }
  }
}

// optional: dump exact input for compare_ort.py --input-npy
void dump_input_nchw() {
  Serial.println("INPUT_DUMP_BEGIN");
  for (int c = 0; c < 3; ++c)
    for (int h = 0; h < IN_H; ++h)
      for (int w = 0; w < IN_W; ++w)
        Serial.printf("%.6f\n", MODEL_INPUT[c][h][w]);
  Serial.println("INPUT_DUMP_END");
  Serial.flush();
}

// ---------------- parse end2end output [300][6] ----------------
// NMS is already done inside the model — just filter by score.
void parse_and_print() {
  // MODEL_OUTPUT is float[300][6]; index [d][c].
  float (*out)[DET_STRIDE] = (float(*)[DET_STRIDE]) MODEL_OUTPUT;
  int shown = 0;
  for (int d = 0; d < NUM_DET; ++d) {
    float x1    = out[d][0];
    float y1    = out[d][1];
    float x2    = out[d][2];
    float y2    = out[d][3];
    float score = out[d][4];
    int   cls   = (int)out[d][5];
    if (score < SCORE_THRESH) continue;   // padding / low-conf
    float w = x2 - x1, h = y2 - y1;
    Serial.printf("[class %d  x=%.1f y=%.1f w=%.1f h=%.1f score=%.3f]\n",
                  cls, x1, y1, w, h, score);
    ++shown;
  }
  Serial.printf("detections: %d\n", shown);
}

// ---------------- optional: dump raw output for ORT comparison ----------------
void dump_output() {
  float (*out)[DET_STRIDE] = (float(*)[DET_STRIDE]) MODEL_OUTPUT;
  Serial.println("OUTPUT_DUMP_BEGIN");
  for (int d = 0; d < NUM_DET; ++d)
    Serial.printf("%.4f %.4f %.4f %.4f %.4f %.4f\n",
      out[d][0], out[d][1], out[d][2], out[d][3], out[d][4], out[d][5]);
  Serial.println("OUTPUT_DUMP_END");
  Serial.flush();
}

void setup(){
  Serial.begin(115200);
  delay(1500);

  if(!psramInit()){ Serial.println("PSRAM init failed!"); return; }
  Serial.printf("Free PSRAM: %u B\n", (unsigned)ESP.getFreePsram());

  if(activations_init()){ Serial.println("activations_init failed (PSRAM arena)"); return; }

  // ---- input: 224x224 RGB888 packed array -> normalized NCHW ----
  fill_input_rgb888(image_1);
  // dump_input_nchw();   // uncomment to capture input for ORT diff

  // ---- run inference ----
  Serial.println("running model...");
  unsigned long t0=micros();
  run_model();
  unsigned long t1=micros();
  Serial.printf("Execution time: %lu us  (%.2f ms)\n", t1-t0, (t1-t0)/1000.0);

  // ---- output ----
  // dump_output();       // uncomment to capture [300][6] for ORT diff
  parse_and_print();

  activations_free();     // optional: free arena if not running again
  Serial.println("done.");
}

void loop(){ delay(1000); }
