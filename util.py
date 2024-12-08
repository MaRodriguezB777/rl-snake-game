import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores, 'bo', label='Score')
    plt.plot(mean_scores, 'gs', label='Mean Score')
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.show(block=False)
    plt.pause(.1)

