#from visdom import Visdom
import matplotlib.pyplot as plt
import numpy as np
import torch
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def visualize(x_train, y_train, x_test, y_test, y_pred, sigma_pred, nll, mse, e,filename=False):
    fig = plt.figure(figsize=(8, 4))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xlabel('X', fontsize=18)
    ax.set_ylabel('Y', fontsize=18)
    ax.set_ylim(-8,8)
#    ax.set_title('Epoch: {}'.format(e), fontsize=18)
    ax.scatter(x_train, y_train, c='g', alpha=0.5)
    ax.scatter(x_test, y_test, c='c', alpha=0.5)
    ax.plot(x_test, y_pred, c='k')
    ax.fill_between(x_test, y_pred - sigma_pred, y_pred + sigma_pred, facecolor='dodgerblue', alpha=0.7)
    ax.fill_between(x_test, y_pred - 2*sigma_pred, y_pred + 2*sigma_pred, facecolor='dodgerblue', alpha=0.5)
    ax.fill_between(x_test, y_pred - 3*sigma_pred, y_pred + 3*sigma_pred, facecolor='dodgerblue', alpha=0.2)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)

#    ax.text(-2, 7, 'NLL: {:.2f} | MSE: {:.2f}'.format(nll, mse), bbox=dict(facecolor='white'), fontsize=18)
    fig.canvas.draw()

    if filename:
        fig.savefig('figs/' + filename, bbox_inches='tight', dpi=300)
    else:
        #Output rasterized plot for doing whatever with
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

def visualize_data_only(x_train, y_train, x_test, y_test, filename='data.pdf'):
    fig = plt.figure(figsize=(8, 4))
    fig.clf()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xlabel('X', fontsize=18)
    ax.set_ylabel('Y', fontsize=18)
    ax.set_ylim(-8,8)
    ax.scatter(x_train, y_train, c='g', alpha=0.5)
    ax.scatter(x_test, y_test, c='c', alpha=0.5)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    fig.canvas.draw()
    ax.legend(['Train','Test'], fontsize=18)
    fig.savefig('figs/' + filename, bbox_inches='tight', dpi=300)

def boxplot(data, title='', legend=None, positions=None, save_path=None):
    plt.figure()
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xtick_loc = []
    for bp, p in zip(data,positions):
        plot = plt.boxplot(bp,vert=True, positions=p)
        xtick_loc.append(p[int((len(p))/2)])
        plt.setp(plot['boxes'][0],color='lightcoral')
        plt.setp(plot['boxes'][1],color='indianred')
        plt.setp(plot['boxes'][2],color='brown')
        plt.setp(plot['boxes'][3],color='firebrick')
#        plot['boxes'][0].set_fillstyle('full')
#        plot['boxes'][0].set_markerfacecolor('lightcoral')


#    plt.ylim(10^0,10^2)
    print(xtick_loc)
    plt.title(title)
    plt.grid()
    plt.xlim(0,p[-1]+1)
    plt.xticks(xtick_loc, legend)
#    plt.xticks(rotation=20)
    plt.yscale('log')
    
    h1, = plt.plot([1,1],'lightcoral')
    h2, = plt.plot([1,1],'indianred')
    h3, = plt.plot([1,1],'brown')
    h4, = plt.plot([1,1],'firebrick')
    plt.legend((h1, h2, h3, h4),('0', '0.01', '0.05', '0.25'))
    h1.set_visible(False)
    h2.set_visible(False)  
    h3.set_visible(False) 
    h4.set_visible(False) 
    
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=800, bbox_inches='tight')