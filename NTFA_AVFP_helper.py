import glob,os
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import permutation_test_score
from sklearn.metrics import make_scorer

# for plotting
def format_plot_style(fig, task_type, plot_sub):
    fig.update_layout(showlegend=False,
                  height=500, width=600, 
                  title_text=task_type + ' ' + str(plot_sub), title_x=0.5,
                  title_font_color="black", title_font_size=28,                 
                  plot_bgcolor="#FFF",
                  xaxis=dict(linecolor='black',mirror=True,linewidth=2,
                             tickfont=dict(size=20, color='black')),
                  yaxis=dict(linecolor='black',mirror=True,linewidth=2,
                             tickfont=dict(size=20, color='black')),
                 )
    return fig

def plot_embeddings(embeddings, hue_var, label_vars=[], marker=16):
    if 'participant' in embeddings.columns:
        hover_vars = ['participant',hue_var]
    else: hover_vars = [hue_var]
    hover_vars.extend(label_vars)
    
    if 'z' not in embeddings.columns:
        fig = px.scatter(embeddings, x='x', y='y',
                         hover_data=hover_vars,color=hue_var)
        fig.update_layout(height=500, width=600,               
                  xaxis=dict(linecolor='black',mirror=True,linewidth=2,
                             tickfont=dict(size=16, color='black'),
                             titlefont=dict(size=20)),
                  yaxis=dict(linecolor='black',mirror=True,linewidth=2,
                             tickfont=dict(size=16, color='black'),
                             titlefont=dict(size=20))
                 )
    else:
        fig = px.scatter_3d(embeddings, x='x', y='y', z='z',
                         hover_data=hover_vars,color=hue_var)
        fig.update_layout(height=500, width=700, 
          scene_aspectmode='cube',
          scene = dict(
              xaxis=dict(linecolor='black',mirror=True,linewidth=2,
                         tickfont=dict(size=16, color='black'),
                         titlefont=dict(size=20)),
              yaxis=dict(linecolor='black',mirror=True,linewidth=2,
                         tickfont=dict(size=16, color='black'),
                         titlefont=dict(size=20)),
              zaxis=dict(linecolor='black',mirror=True,linewidth=2,
                         tickfont=dict(size=16, color='black'),
                         titlefont=dict(size=20)),
          ), margin=dict(l=0.1, r=0.1, b=0.1, t=0.1)
         )
    fig.update_traces(marker=dict(size=marker))
    #fig.show()
    return fig    

# data manupulation
def get_video_category(vn):
    if vn in [1,3,5,8,10,44,2,4,9,12,39,48]:
        cat = 'Heights'
    elif vn in [13,20,21,22,49,60,16,24,50,51,58,59]:
        cat = 'Social'
    elif vn in [29,33,35,61,64,72,32,36,65,69,70,71]:
        cat = 'Spiders'
    else:
        raise Exception('no this video number!')
    return cat 


def median_labeler(data, median):
    if data < median:
        med = 'Low'
    else:
        med = 'High'
    return med


def parse_task_lines(lines, headers, median_type, median_value_list):
    for (i, line) in enumerate(lines):
        cols = line.split(' ')
        
        task = cols[int(np.where(headers == 'video_name')[0])]
        mem = cols[int(np.where(headers == 'novel_vs_familiar')[0])]
        study = 'AVFP'

        if mem == '1':
            mem_type = 'New'
        else: mem_type = 'Old'
        task = f'{mem_type}_{task[:-4]}_{study}'
        
        video_category = cols[int(np.where(headers == 'video_category')[0])]
        video_category_index = int(video_category)-1
        median = median_value_list[video_category_index]
        
        start_time = float(cols[int(np.where(headers == 'video_onset')[0])])
        end_time = float(cols[int(np.where(headers == 'video_offset')[0])])
        run = int(cols[int(np.where(headers == 'run_number')[0])])
        video_number = int(cols[int(np.where(headers == 'video_number')[0])])
        fear_rating = abs(float(cols[int(np.where(headers == 'fear_rating')[0])]))
        arousal_rating = abs(float(cols[int(np.where(headers == 'arousal_rating')[0])]))
        valence_rating = abs(float(cols[int(np.where(headers == 'valence_rating')[0])]))
        
        if median_type == 'fear':
            if fear_rating > median:
                task_cat = task + '_High' + '_' + median_type
            else:
                task_cat = task + '_Low' + '_' + median_type
        
        elif median_type == 'arousal':  
            if arousal_rating > median:
                task_cat = task + '_High' + '_' + median_type
            else:
                task_cat = task + '_Low' + '_' + median_type
        
        elif median_type == 'valence':
            if valence_rating > median:
                task_cat = task + '_High' + '_' + median_type
            else:
                task_cat = task + '_Low' + '_' + median_type
            
            
        if np.isnan(fear_rating): # if didn't move the slider at all
            fear_rating = .5 #middle
        yield [task, start_time, end_time, run, video_number, fear_rating, task_cat]

        
def load_sub_log_info(subjects,log_file_headers,log_path):
    '''
    input:
    subjects: a list of subjects to be included in the 
    log_file_headers: np.array of the column name list of the logfiles
    log_path: str, logfiles directory
    
    output:
    log_df: a list of logfiile information including stimulus name, onset, offset, run, video_number, raw fear rating, stimulus name with median split 
        (task_lines[s] index the subject)
    '''
    log_files = []
    for s in subjects:
        log_files.append(glob.glob(log_path + f'*{s}*.txt'))


    task_lines =[]
    
    for task_csv in log_files: 
        task_csv = task_csv[0]

        subject = int(task_csv.split('.txt')[0][-3:])
        # print("loading task_csv from subjects: ", subject)
        median_type = 'fear'
        df = pd.DataFrame(np.loadtxt(task_csv,dtype =str))
        col_index = np.where(log_file_headers == median_type +'_rating')[0][0]
        median_value = []
        for situation in ["1","2","3"]:
            row_index = df[2]==situation
            median_value.append(df.loc[row_index,col_index].astype(float).abs().median())
        with open(task_csv, 'r') as task_csv_file:
            task_lines.append(list(parse_task_lines(task_csv_file.readlines(), log_file_headers, median_type,median_value)))
            
    return task_lines         
    
    

# extract embeddings:    
def fetch_embeddings_v1(): 
    hyperparams = dtfa.variational.hyperparams.state_vardict()
    tasks = dtfa.tasks()
    subjects = dtfa.subjects()
    z_p_mu = hyperparams['subject_weight']['mu'].data
    z_s_mu = hyperparams['task']['mu'].data

    z_ps_mu, combinations = list(), list()
    for p in range(len(subjects)):
        # because I coded by memory, participants only have 1/2 of the unqiue tasks each - find index:
        sub_tasks = [b['task'] for b in avfp_db.blocks.values() if b['subject'] == subjects[p]]
        combinations.append(np.vstack([np.repeat(subjects[p],len(sub_tasks)), np.array(sub_tasks)]))
        for t in range(len(sub_tasks)):
            task_index = [i for i, e in enumerate(tasks) if e == sub_tasks[t]]
            joint_embed = torch.cat((z_p_mu[p], z_s_mu[task_index[0]]), dim=-1)
            interaction_embed = dtfa.decoder.interaction_embedding(joint_embed).data
            z_ps_mu.append(interaction_embed.data.numpy())
    z_ps_mu = np.vstack(z_ps_mu)   
    combinations = np.hstack(combinations).T  

    # convert to dataframes
    z_p = pd.DataFrame(np.hstack([np.reshape(subjects, (len(subjects),1)), z_p_mu.numpy()]),
                       columns=['participant','x','y'])
    z_s = pd.DataFrame(np.hstack([np.reshape(tasks, (len(tasks),1)), z_s_mu.numpy()]),
                       columns=['stimulus','x','y'])
    z_ps = pd.DataFrame(np.hstack([combinations, z_ps_mu]),
                        columns=['participant','stimulus','x','y'])
    return z_p, z_s, z_ps


def fetch_embeddings_v2(): 
    hyperparams = dtfa.variational.hyperparams.state_vardict()
    tasks = dtfa.tasks()
    subjects = dtfa.subjects()
    interactions = dtfa._interactions
    z_p_mu = hyperparams['subject_weight']['mu'].data
    z_s_mu = hyperparams['task']['mu'].data
    z_i_mu = hyperparams['interaction']['mu'].data
    
    z_p_sigma = torch.exp(hyperparams['subject_weight']['log_sigma'].data)
    z_s_sigma = torch.exp(hyperparams['task']['log_sigma'].data)
    z_i_sigma = torch.exp(hyperparams['interaction']['log_sigma'].data)

    # convert to dataframes
    z_p = pd.DataFrame(np.hstack([np.reshape(subjects, (len(subjects),1)), z_p_mu.numpy(), z_p_sigma.numpy()]),
                       columns=['participant','x','y', 'x_sigma','y_sigma'])
    z_s = pd.DataFrame(np.hstack([np.reshape(tasks, (len(tasks),1)), z_s_mu.numpy(), z_s_sigma.numpy()]),
                       columns=['stimulus','x','y', 'x_sigma', 'y_sigma'])
    z_ps = pd.DataFrame(np.hstack([interactions, z_i_mu.numpy(), z_i_sigma.numpy()]),
                        columns=['participant','stimulus','x','y', 'x_sigma', 'y_sigma'])
    return z_p, z_s, z_ps



# for modeling
def GetEmbeddingXY(embedding, s,cv_column, x = 'psc', which_y = 'fear_rating'):
    x_columns = []
    if 's' in x:
        x_columns.append('s_x')
        x_columns.append('s_y')
    if 'c' in x:
        x_columns.append('c_x')
        x_columns.append('c_y')
    if 'p' in x:
        x_columns.append('p_x')
        x_columns.append('p_y')
    
    
    if cv_column == 'all':
        X = embedding.loc[:, x_columns].values
        Y = embedding.loc[:, which_y].values
    else: 
    
        X = embedding.loc[embedding[cv_column].isin(s), x_columns].values
        Y = embedding.loc[embedding[cv_column].isin(s), which_y].values
    
    return X, Y

def TrainEmbeddingModel(model, kf, data_ids, groups, cv_column, embedding_choice, which_y, embedding_df, acc = 'corr', verbose = True):
    test_acc_list = []
    test_rmse_list = []
    for i, (train_idx, test_idx) in enumerate(kf.split(data_ids, groups)):
        

        data_train = np.array(data_ids)[train_idx.astype(int)].tolist()
        data_test = np.array(data_ids)[test_idx.astype(int)].tolist()

        X_train, y_train = GetEmbeddingXY(embedding_df, data_train, cv_column, x=embedding_choice, which_y = which_y)
        X_test, y_test = GetEmbeddingXY(embedding_df, data_test, cv_column, x=embedding_choice, which_y = which_y)

        model.fit(X_train,y_train)

        y_test_pred = model.predict(X_test)
        if acc == 'corr':
            test_acc = np.corrcoef(y_test,y_test_pred)[0,1]
        else: 
            test_acc = acc(y_test, y_test_pred)
            
        test_acc_list.append(test_acc)
        
        rmse = sqrt(mean_squared_error(y_test,y_test_pred))
        test_rmse_list.append(rmse)
        if verbose:
            print(f'---------------------- fold {i+1} ---------------------------')
            print(test_acc)
    
    print(f'mean accuracy = {np.mean(test_acc_list)}')
    print(f'mean rmse = {np.mean(test_rmse_list)} \n')
    
    return test_acc_list, test_rmse_list, model



def MyAccFunc(y_true, y_pred):
    acc = np.corrcoef(y_true, y_pred)[0,1]
    return acc



def PermutationP(permutation_scores, score, n_permutations):
    pvalue = (np.sum(permutation_scores >= score) + 1.0) / (n_permutations + 1)
    return pvalue


def RunPermutation(score, model, group_column, embedding_choice, embedding_df, n_split = 6, which_y = 'fear', n_permutation = 100):

    X, Y = GetEmbeddingXY(embedding_df, None, 'all', x=embedding_choice, which_y = which_y)
    if group_column:
        _, perm_scores, _ = permutation_test_score(
            model, X, Y, groups = embedding_df[group_column], scoring=make_scorer(MyAccFunc), cv=n_split, n_permutations=n_permutation)
    else:
        _, perm_scores, _ = permutation_test_score(
            model, X, Y, groups = None, scoring=make_scorer(MyAccFunc), cv=n_split, n_permutations=n_permutation)

    pvalue = PermutationP(perm_scores, score, n_permutation)
    
    fig, ax = plt.subplots()

    ax.hist(perm_scores, bins=20, density=True)
    ax.axvline(score, ls="--", color="r")
    ax.set_title(f"Score: {score:.3f}, p-value: {pvalue:.3f}")
    ax.set_xlabel("Pearson's R")
    _ = ax.set_ylabel("Frequency")
    ax.set_xlim([-0.1, 0.65])
    ax.set_ylim([0, 25])
    
    
    return pvalue, perm_scores