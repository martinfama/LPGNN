import matplotlib.pyplot as plt

def plot_losses_vs_epochs(pr_dict, save_name=None):
    """ Plot the losses vs epochs. 
    
    Args:
        pr_dict (dict): Dictionary of PR result, including epochs and losses.

    Returns:
        plt.figure: The figure of the plot.
    """
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid(alpha=0.5)

    for key, loss in pr_dict['losses'].items():
        if key != 'epochs' and key != 'test loss':
            ax.plot(pr_dict['losses']['epochs'], pr_dict['losses'][key], label=f'Loss {key}')
    
    ax.legend(loc='upper right')

    if save_name is not None:
        plt.savefig(save_name)
    
    return fig