import gymnasium as gym
import miniworld
from PIL import Image

# (1) create the environment. (render_mode=rgb_array)
env = gym.make("MiniWorld-Custom-v0", render_mode="rgb_array", view="top_local")

# (2) reset the envrironment.
env.reset()

# (3) render the image.
image = env.render()
print(f"Image shape: {image.shape}")

# (4) save the image.
im = Image.fromarray(image)
im.save("render_test.png")

# centric_top_image = env.render_local_top_view()
# print(f"Centric top image shape: {centric_top_image.shape}")
# im = Image.fromarray(centric_top_image)
# im.save("centric_top_image.png")
