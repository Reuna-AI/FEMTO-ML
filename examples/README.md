# Examples

This folder is for example sketches that run a Synapedge-generated model on-device.

A typical ESP32 sketch:

1. includes the generated `model.h` and calls `activations_init()` then `run_model()`,
2. includes the esp-nn SIMD kernel sources (`esp_nn_conv_s8_simd.*`,
   `esp_nn_depthwise_conv_s8_simd.*`) if using the `esp32s3 --simd` backend,
3. feeds a preprocessed int8 input (e.g. 256×256, NCHW) and prints the detections.

See the top-level [README](../README.md) for the compile command that generates the model
sources.

---

## Board setup — ESP32-S3 N16R8 (16 MB flash, 8 MB PSRAM)

Synapedge models are large (weights + activations), so a generated project usually needs a
**custom partition table** that gives most of the flash to the application. The steps below
add a 15 MB-APP partition scheme for the ESP32-S3 N16R8 under the Arduino ESP32 core.

> Paths below are for Arduino ESP32 core **2.0.11** on Windows. Adjust the version number and,
> on macOS/Linux, the base path:
> - Windows: `C:\Users\<your_username>\AppData\Local\Arduino15\packages\esp32\hardware\esp32\2.0.11\`
> - macOS:   `~/Library/Arduino15/packages/esp32/hardware/esp32/2.0.11/`
> - Linux:   `~/.arduino15/packages/esp32/hardware/esp32/2.0.11/`

### 1. Edit `boards.txt`

Locate the `boards.txt` file for the ESP32 package, for example:

```
C:\Users\<your_username>\AppData\Local\Arduino15\packages\esp32\hardware\esp32\2.0.11\boards.txt
```

Replace `<your_username>` with your actual username, open the file in a text editor, and find
the section beginning with `esp32s3.menu.PartitionScheme`. At the **end** of that section, add:

```
esp32s3.menu.PartitionScheme.My_16MB=16M Flash (15MB APP)
esp32s3.menu.PartitionScheme.My_16MB.build.partitions=My_16MB
esp32s3.menu.PartitionScheme.My_16MB.upload.maximum_size=15728640
```

### 2. Create the partition table

Create a file named `My_16MB.csv` with the following contents:

```
# Name,      Type,   SubType,  Offset,   Size,       Flags
nvs,        data,   nvs,      0x9000,   0x5000,
otadata,    data,   ota,      0xe000,   0x2000,
app,        app,    factory,  0x10000,  0xF00000,
ffat,       data,   fat,      0xF10000, 0xE0000,
coredump,   data,   coredump, 0xFF0000, 0x10000,
```

This lays out the 16 MB flash as: a ~15 MB application partition (`0xF00000`), a FAT data
region, and small NVS / OTA-data / coredump regions.

Save `My_16MB.csv` in the ESP32 partitions folder:

```
C:\Users\<your_username>\AppData\Local\Arduino15\packages\esp32\hardware\esp32\2.0.11\tools\partitions
```

Replace `<your_username>` with your actual username.

### 3. Select the scheme in the Arduino IDE

1. Restart the Arduino IDE (so it re-reads `boards.txt`).
2. Select your ESP32-S3 board under **Tools → Board**.
3. Set **Tools → PSRAM** to **OPI PSRAM** (required for the N16R8's 8 MB PSRAM, which the
   arena uses for large activation/weight buffers).
4. Set **Tools → Partition Scheme** to **"16M Flash (15MB APP)"** (the entry you just added).
5. Set **Tools → Flash Size** to **16MB (128Mb)**.

You can now build and flash a Synapedge-generated sketch that uses the full 16 MB flash and
8 MB PSRAM of the N16R8.

> **Note:** these steps modify the Arduino ESP32 core in place. Updating or reinstalling the
> core will overwrite `boards.txt` and remove `My_16MB.csv`, so you may need to re-apply them
> after a core update.