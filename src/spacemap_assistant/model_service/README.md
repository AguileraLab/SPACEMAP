# Local model service

This package contains the small server that connects the SPACEMAP assistant to
a model running on your own machine. It is started and stopped from the GUI;
you should not normally need to run it by hand.

The server:

- listens only on loopback, on a port you can change in the assistant settings;
- loads whichever model directory it is given on the command line, so any
  catalog entry can be served by the same code;
- works fully offline once the weights are downloaded;
- runs on a CUDA GPU when one is present, and on the CPU in float32 when it is
  not — slower, but functional;
- picks bfloat16, 8-bit or 4-bit automatically to fit the available GPU memory;
- accepts text, and uploaded PNG/JPEG images when the model is multimodal;
- falls back to a text-only path for models that cannot read images;
- blocks remote image URLs and exposes no shell or filesystem tools;
- requires the private key created in the assistant configuration directory.

Start it from **AI Assistant → Model** after selecting and downloading a model
in **Tab 0 → AI Assistant**. The first start can take several minutes while the
weights are read into memory.

Run it directly only for debugging:

```bash
# Run the file directly. The application launches it this way too, so the
# server needs only torch, transformers and fastapi -- not the full
# scientific stack that importing the package would pull in.
python "$(python -c 'import spacemap_assistant.model_service as m; print(m.__path__[0])')/server.py" \
    --model-dir ~/.spacemap/models/google/gemma-4-12B-it \
    --model-id google/gemma-4-12B-it \
    --port 8091 \
    --api-key-file ~/.spacemap/assistant_data/config/local_model.key
```

The model's opinion about an image is advisory. Cell labels, tissue rejection,
blur, artifacts and segmentation quality still need a human or deterministic QC
check.
