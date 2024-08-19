from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")


prompt = "A beautiful view of the ocean from a cruise ship."


image = pipe(prompt).images[0]


image.save("generated_image.png")

image.show()
