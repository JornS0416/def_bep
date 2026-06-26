import imageio

def save_video(frames, output_path, fps=30):
    imageio.mimsave(output_path, frames, fps=fps)