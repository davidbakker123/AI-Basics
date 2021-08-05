import numpy as np
import matplotlib.pyplot as plt

hbar = 1
w0 = 0.5
e = 1
m_star = 1

VS, CS = 1, 1
VD, CD = 1, 1
VG, CG = 1, 1
C = CS + CD + CG

def E(n, l, B):
    wc = e * B / m_star
    w_0 = w0 * np.ones(len(B))
    return (2 * n + np.abs(l) + 1 ) * np.sqrt(w0**2 + (1/4) * wc ** 2) - (1/2) * l * w0

def U(N):
    return (-e * N + CS * VS + CD * VD + CG * VG) ** 2 / (2 * C)

def plot_spectrum():
    B_size = 200
    B = np.linspace(0, 3, B_size)
    energy = np.zeros([9, B_size])
    energy[0, :] = E(0, 0, B)
    energy[1, :] = E(0, 1, B)
    energy[2, :] = E(0, 2, B)
    energy[3, :] = E(0, 3, B)
    energy[4, :] = E(0, 4, B)
    energy[5, :] = E(0,-1, B)
    energy[6, :] = E(0,-2, B)
    energy[7, :] = E(1, 0, B)
    energy[8, :] = E(1,-1, B)

    labels = ["(0,0)", "(0,1)", "(0,2)", "(0,3)", 
    "(0,4)", "(0,-1)", "(0,-2)", "(1,0)", "(1,-1)"]

    for i in range(9):
        plt.plot(B, energy[i, :], 'k')
        x, y = B[-1], energy[i, -1]
        plt.annotate(labels[i], # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(-10,5), # distance from text to points (x,y)
                    ha='center')
    plt.show()

from matplotlib.widgets import Slider, Button

# The parametrized function to be plotted
def f(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

t = np.linspace(0, 1, 1000)

# Define initial parameters
init_amplitude = 5
init_frequency = 3

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line, = plt.plot(t, f(t, init_amplitude, init_frequency), lw=2)
ax.set_xlabel('Time [s]')

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = plt.axes([0.15, 0.25, 0.0225, 0.63], facecolor=axcolor) #[0.25, 0.1, 0.65, 0.03], 
freq_slider = Slider(
    ax=axfreq,
    label='f [Hz]',
    valmin=0.1,
    valmax=30,
    valinit=init_frequency,
    orientation="vertical"
)

# Make a vertically oriented slider to control the amplitude
axamp = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
amp_slider = Slider(
    ax=axamp,
    label="A",
    valmin=0,
    valmax=10,
    valinit=init_amplitude,
    orientation="vertical"
)

# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()