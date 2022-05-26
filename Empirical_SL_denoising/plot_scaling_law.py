import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle

def lin_fit(x,y,start_x,end_x):
    # train_size at which power-law regime begins
    start_ind = np.where(x==start_x)[0][0]
    # train_size at which power-law regime ends
    end_ind = np.where(x==end_x)[0][0]

    # find linear fit 
    y_fit = np.log10(y[start_ind:end_ind+1])
    x_fit = np.vstack((np.log10(x[start_ind:end_ind+1]),np.ones(y_fit.shape))).T

    linfit_w = np.linalg.inv(x_fit.T@x_fit) @ x_fit.T@y_fit

    # power-law: R = beta * N**alpha
    beta = 10**linfit_w[1]
    alpha = linfit_w[0]
    print(beta,alpha)
    linfit = beta * x**alpha
    return linfit, alpha, beta

    

def plot_trainsize_scaling(ax,best_perf_dict,train_examples_tags,dist_shifts,fontsize,linfit_bounds,colors,
                          ylim=None,xlim=None):
    
    
    for j,dist_shift in enumerate(dist_shifts):  
        x_points = []
        y_points = []
        x_line = []
        y_line = []
        for tag in train_examples_tags:
            x_line.append(int(tag[1:-1]))
            y_line.append(best_perf_dict[tag][dist_shift]['max'])

            for i in range(len(best_perf_dict[tag][dist_shift]['all'])):
                x_points.append(int(tag[1:-1]))
                y_points.append(best_perf_dict[tag][dist_shift]['all'][i])
        linfits = []
        alphas = []
        if linfit_bounds:
            if not isinstance(linfit_bounds[0],list):
                linfit, alpha, _ = lin_fit(np.array(x_line),np.array(y_line),linfit_bounds[0],linfit_bounds[1])
            else:
                for linfit_bound in linfit_bounds:
                    linfit, alpha, _ = lin_fit(np.array(x_line),np.array(y_line),linfit_bound[0],linfit_bound[1])
                    linfits.append(linfit)
                    alphas.append(alpha)
        else:
            alpha=0

        if dist_shift == 'test_':
            label = r"ImgNet Test"
            #label = 'test'
        else:
            label = r"%s: $\alpha={%.4f}$"%(dist_shift,np.round(alpha,4))
            #label = 'test'

        ax.plot(x_line,y_line,label=label,color=colors[j])
        if linfit_bounds:
            if linfits:
                for k,(linfit,alpha) in enumerate(zip(linfits,alphas)):
                    label = r"$\alpha={%.4f}$"%(np.round(alpha,4))
                    ax.plot(x_line,linfit,linestyle='--',label=label,color=colors[j+k+1])
            else:
                label = r"$\alpha={%.4f}$"%(np.round(alpha,4))
                ax.plot(x_line,linfit,linestyle='--',label=label,color=colors[j])
        ax.scatter(x_points,y_points,color=colors[j])

    
    
    ax.legend(fontsize=fontsize-3)
    ax.set_xlabel("Number of exsamples in the training set $N$", fontsize=fontsize)
    ax.set_ylabel("PSNR (dB)", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-2)
    ax.set_xscale('log')
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    ax.grid(True)
    
def plot_parameter_scaling(ax,perf_dict,train_examples_tags,dist_shifts,fontsize,colors,ylim=None,xlim=None):
    layers_channels = ['l2c16','l2c32','l2c64','l2c128','l2c192','l2c256','l2c320']
    num_parameters = [116707,465859,1861507,7442179,16742019,29761027,46499203]
        
    for j,dist_shift in enumerate(dist_shifts):   
        
        for c,tag in enumerate(train_examples_tags):
            num_train = int(tag[1:-1])
            num_train = int(tag[1:-1])
            x_points = []
            y_points = []
            x_line = []
            y_line = []
            experiments = perf_dict[tag].keys()

            for exp in experiments:
                use_best_or_last = 'best'
                eind = exp.find('_l')+1
                # Get the number of parameters
                lc_tag = exp[eind:exp.find('_bs')]
                for k,lc in enumerate(layers_channels):
                    if lc == lc_tag:
                        num_par = num_parameters[k]
                # Get the number of runs for this parameter count
                num_runs = len(perf_dict[tag][exp][dist_shift]['best'])

                x_line.append(num_par)
                y_line.append(np.max(perf_dict[tag][exp][dist_shift][use_best_or_last]))
                        
                for i in range(num_runs):
                    y_points.append(perf_dict[tag][exp][dist_shift][use_best_or_last][i])
                    x_points.append(num_par)           
            
            ax.scatter(x_points,y_points,color=colors[c])
            label = r"$N%i$"%(num_train)
            ax.plot(x_line,y_line,label=label,color=colors[c])
        
            

    ax.legend(fontsize=fontsize-4)
    ax.set_xlabel("Number of network parameters $P$", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-2)
    ax.set_xscale('log')
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    ax.grid(True)
    
def generate_performance_dicts(dist_shifts,train_examples_tags,dist_exps_list):
    if "val_" in dist_shifts:
        pass
    else:
        raise ValueError("val_ metrics must be included to determine best experiment per setup") 


    distinct_nums = []
    exp_num_to_train_size = {}
    for tag in train_examples_tags:
        exp_num_to_train_size[tag] = []
        for dist_exp in dist_exps_list:
            if tag in dist_exp:
                if not any(distinct_num in dist_exp for distinct_num in distinct_nums):
                    exp_num_to_train_size[tag].append(dist_exp)
                    distinct_nums.append(dist_exp)

    all_exps_list = glob.glob('E*')

    perf_dict = {}
    for tag in train_examples_tags:
        perf_dict[tag] = {}
        for exp_num in exp_num_to_train_size[tag]:
            perf_dict[tag][exp_num] = {}
            for dist_shift in dist_shifts:
                perf_dict[tag][exp_num][dist_shift] = {}
                for ckpt in ['best','last']:
                    perf_dict[tag][exp_num][dist_shift][ckpt] = []

    for exp in all_exps_list:
        metrics_list = glob.glob(exp+'/*metrics*')

        for dist_exp in dist_exps_list:
            if exp==dist_exp: 
                if 'run' in dist_exp:
                    eind = dist_exp.find('_run')
                    exp_num = dist_exp[0:eind]
                else:
                    exp_num = dist_exp
                for tag in train_examples_tags: 
                    if tag in exp: 
                        
                        checkpoint_path = glob.glob(exp +'/unet*')
                        if len(checkpoint_path) != 1:
                            raise ValueError("There is either no or more than one model to load events from")
                        for dist_shift in dist_shifts: 
                            ckpt = 'best'
                            for metric in metrics_list:
                                if dist_shift in metric and ckpt in metric: # find the correct metric file        
                                    metrics_dict = pickle.load( open( metric, "rb" ) )
                                    perf_dict[tag][exp_num][dist_shift][ckpt].append(metrics_dict[f'psnr_m'][0])

    # Get best mean/std or median performance per trainset size
    print('Mean/std performance:')
    best_perf_dict = {}
    for tag in train_examples_tags:
        best_perf_dict[tag] = {}
        for dist_shift in dist_shifts:
            best_perf_dict[tag][dist_shift] = {}
            best_perf_dict[tag][dist_shift]['mean'] = 0
            best_perf_dict[tag][dist_shift]['max'] = 0
            best_perf_dict[tag][dist_shift]['std'] = 0
            best_perf_dict[tag][dist_shift]['median'] = 0
            best_perf_dict[tag][dist_shift]['all'] = []


    for dist_shift in dist_shifts:
        print('\n')
        for tag in train_examples_tags:
            best_per_exp_num = []
            exp_nums = []
            print('''{} {} all experiments:'''.format(dist_shift,tag))
            for exp_num in exp_num_to_train_size[tag]:
                use_best_or_last = 'best'

                exp_nums.append(exp_num)
                best_per_exp_num.append(np.max(perf_dict[tag][exp_num]["val_"][use_best_or_last])) #only use val metric to compare experiments
                
                print_all_psnr = [np.round(tt,4) for tt in perf_dict[tag][exp_num][dist_shift][use_best_or_last]]
                print('''{} with PSNR mean {} max {} std {} all {}\n'''.format(exp_num,
                                                                            np.round(np.mean(perf_dict[tag][exp_num][dist_shift][use_best_or_last]),4),
                                                                             np.round(np.max(perf_dict[tag][exp_num][dist_shift][use_best_or_last]),4),
                                                                            np.round(np.std(perf_dict[tag][exp_num][dist_shift][use_best_or_last]),4),
                                                                            print_all_psnr                                                  
                                                                            ))

            ind = np.where(best_per_exp_num==np.max(best_per_exp_num))[0][0]
            best_exp_num = exp_nums[ind]
                
            best_perf_dict[tag][dist_shift]['mean'] = np.mean(perf_dict[tag][best_exp_num][dist_shift][use_best_or_last])
            best_perf_dict[tag][dist_shift]['std'] = np.std(perf_dict[tag][best_exp_num][dist_shift][use_best_or_last])
            best_perf_dict[tag][dist_shift]['median'] = np.median(perf_dict[tag][best_exp_num][dist_shift][use_best_or_last])
            best_perf_dict[tag][dist_shift]['all'] = perf_dict[tag][best_exp_num][dist_shift][use_best_or_last]   
            best_perf_dict[tag][dist_shift]['max'] = np.max(perf_dict[tag][best_exp_num][dist_shift][use_best_or_last])   
            print('best experiment: {} with PSNR mean {} max {} std {} \n'.format(best_exp_num,
                                                                                        np.round(best_perf_dict[tag][dist_shift]['mean'],4),
                                                                                  np.round(best_perf_dict[tag][dist_shift]['max'],4),
                                                                                       np.round(best_perf_dict[tag][dist_shift]['std'],4)))
    return best_perf_dict, perf_dict


def run():
    dist_exps_list = glob.glob('E0*')
    dist_exps_list.sort()
    train_examples_tags = []
    for dist_exps in dist_exps_list:
        tag = dist_exps[dist_exps.find('t'):dist_exps.find('_l')+1]
        if tag not in train_examples_tags:
            train_examples_tags.append(tag)

    dist_shifts = ["val_","test_"]
    best_perf_dict, perf_dict = generate_performance_dicts(dist_shifts,train_examples_tags,dist_exps_list)
    
    ######
    # Performance as function of number of training examples
    ######


    fig = plt.figure(figsize=(20,7))
    fontsize = 22
    ax1 = fig.add_subplot(121)
    colors = ['b','r','k','g','m','b','r','k','g','m']
    dist_shifts = ['test_']

    ylim = []
    # Specify from which training set size to which training set size to fit a linear power law
    # e.g. [300,3000] or a list of start and end points to get several power laws, e.g. [[300,3000],[3000,10000]]
    linfit_bounds = [] 

    plot_trainsize_scaling(ax1,best_perf_dict,train_examples_tags,dist_shifts,fontsize,linfit_bounds=linfit_bounds,
                              colors=colors, ylim=ylim,xlim=None)


    ######
    # Performance as function of number of network parameters
    ######
    ax2 = fig.add_subplot(122)
    plot_parameter_scaling(ax2,perf_dict,train_examples_tags,dist_shifts,fontsize,colors,ylim=ylim,xlim=None)

    plt.savefig("Empirical_SL_denoising.png",dpi=150)
 
                
if __name__ == '__main__':
    run()            
    