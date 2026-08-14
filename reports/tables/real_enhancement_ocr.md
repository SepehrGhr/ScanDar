### Enhancement vs CamScanner, on the photos with a reference scan

5 photos rectified with their annotated corners (never the detector's own — this isolates the enhancement network). Tesseract's own mean word confidence (0-100) and word count, identical preprocessing on all three variants.

| Photo | input conf | input words | ours conf | ours words | reference conf | reference words |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image1 | 31 | 237 | 31 | 298 | 30 | 255 |
| Image11 | 35 | 238 | 38 | 239 | 33 | 256 |
| Image18 | 74 | 252 | 73 | 242 | 79 | 243 |
| Image7 | 34 | 124 | 34 | 138 | 31 | 101 |
| Image9 | 32 | 113 | 37 | 145 | 33 | 150 |

**Mean word confidence** — input 41.1, ours 42.5, reference 41.2.

Character/word error rate against a hand-typed transcript:

| Photo | input CER | input WER | ours CER | ours WER | reference CER | reference WER |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image18 | 0.250 | 0.433 | 0.285 | 0.465 | 0.173 | 0.343 |
