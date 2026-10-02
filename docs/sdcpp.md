# stable-diffusion.cpp Runtime & GGUF Integration

## Target Architecture
The model is quantized and converted to GGUF format for real-time offline inference using `stable-diffusion.cpp`.

## Quantization Formats Supported
- **F16**: Master unquantized reference.
- **Q8_0**: High-precision quantization for maximum visual fidelity.
- **Q6_K / Q5_K**: Balanced quantization tiers.
- **Q4_K**: Default edge and mobile deployment quantization.

## Execution Syntax
```cmd
sd-cli.exe -m models/hcs-image-evolver-q4.gguf -p "A majestic lion standing proudly on a savannah rock at sunrise" -o output.png --steps 4 --cfg-scale 1.0 --sampling-method euler
```
