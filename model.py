import pandas as pd


# Note: We have created the user_final_rating.csv using our best model trained in Jupyter Notebook. 
# Refer to section - "Loading the best model" from jupyter notebook
# So we load user_final_rating.csv as a look up table.


# Load user_final_rating.csv

ufr = pd.read_csv('user_final_rating.csv')
ufr.set_index('reviews_username', inplace=True)


def get_products(user_input):
   
    d = pd.DataFrame(ufr.loc[user_input].sort_values(ascending=False)[0:5]).reset_index()

    return_list = d['index'].values
    return list(return_list)

if __name__=='__main__':
    # r = get_products('01impala')
    # print(r)

    with open('index.html', 'r') as f:
        data = f.read()

    print(data)



