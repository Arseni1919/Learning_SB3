import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import gradio as gr
import tempfile

def create_animation():
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'b-', animated=True)

    def init():
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-1.1, 1.1)
        return ln,

    def update(frame):
        xdata.append(frame)
        ydata.append(np.sin(frame))
        ln.set_data(xdata, ydata)
        return ln,

    ani = animation.FuncAnimation(
        fig, update, frames=np.linspace(0, 2*np.pi, 100),
        init_func=init, blit=True, repeat=False
    )

    # Save to MP4
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    ani.save(temp_video.name, writer='ffmpeg', fps=20)
    plt.close(fig)

    return temp_video.name

with gr.Blocks() as demo:
    with gr.Row():
        btn = gr.Button("Generate Animation")
        vid = gr.Video(label="Animated Plot", autoplay=True)

    btn.click(fn=create_animation, outputs=vid)

demo.launch()
