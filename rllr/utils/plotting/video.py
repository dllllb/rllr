import base64
import glob
import io
from IPython.display import HTML
from IPython import display


def show_video(path='./video/', idx=0):
    mp4list = sorted(glob.glob(f'{path}*.mp4'))
    if len(mp4list) > 0:
        mp4 = mp4list[idx]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")
