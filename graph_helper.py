import matplotlib.pyplot as plt

def make_graphs(scores, mean_scores, scores_2, mean_scores_2):
    plt.clf() #clear current figure
    plt.title('Snake')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.subplot(2,1,1)
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    plt.clf() #clear current figure
    plt.title('Viper')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.subplot(2,1,2)
    plt.plot(scores_2)
    plt.plot(mean_scores_2)
    plt.ylim(ymin=0)
    plt.text(len(scores_2)-1, scores_2[-1], str(scores_2[-1]))
    plt.text(len(mean_scores_2)-1, mean_scores_2[-1], str(mean_scores_2[-1]))

    plt.pause(.01)