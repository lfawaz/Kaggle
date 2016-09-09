
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import datetime
import dateparser


def load_clean_data():

    train = pd.read_csv("../data/train.csv")

    print"load data..."
    model_data = train
    ##create hour
    model_data['hour'] = model_data['time'].apply(lambda x: x.split(':')[0])
    ##create minute
    model_data['minute'] = model_data['time'].apply(lambda x: x.split(':')[1])


    ##replace car value, risk_fact,C_Previous,duration_previous null with -1
    print"removing nulls..."
    null_columns = ['car_value','risk_factor','C_previous','duration_previous']
    for col in null_columns:
        model_data[col] = model_data[col].apply(lambda x: -1 if pd.isnull(x) else x)

    ############################################################################################   
    ##implement is_last column this determines what was the last record the customer looked at##
    ############################################################################################
    #Select first two columns for faster processing
    is_last_data = model_data[['customer_ID','shopping_pt']]
    #Set an empty column to is_last
    is_last_data['is_last'] = 0

    #convert the Pandas frame work to numpy because it is faster to loop through
    np_is_last_data = np.asarray(is_last_data)
    print "adding is_last column ..."
    #create a column to indicate if this was the last viewed record
    for i in range(len(np_is_last_data)):
        if np_is_last_data[i][1] == 1:
            np_is_last_data[i - 1][2] = 1

    #create the data frame with the is_last column
    is_last_data = pd.DataFrame(np_is_last_data, columns=is_last_data.columns.values)


    ######################################################################
    #create a flag to determine if the record was the finally sold record#
    ######################################################################

    #outer join data with subset of purchases on all the product items
    print"adding is_final column -predictor-"
    #select the purchased record
    sold_records_only = model_data[['customer_ID','shopping_pt','A','B','C','D','E','F','G']][(model_data.record_type == 1)]
    is_final_merge = pd.merge(model_data[['customer_ID','shopping_pt','A','B','C','D','E','F','G']],sold_records_only,on=['customer_ID','A','B','C','D','E','F','G'], how='outer')

    #lamdba function if the value of shopping_pt_y is null since it is outer join then the production was not 
    #purchased otherwise it was eventually purchase, we will use this column as our predictor
    is_final_merge['is_final'] = is_final_merge['shopping_pt_y'].apply(lambda x: 0 if pd.isnull(x) else 1)
    is_final_merge.rename(columns={'shopping_pt_x':'shopping_pt'}, inplace=True)
    is_final_data = is_final_merge[['customer_ID','shopping_pt','is_final']]


    ###################################################################
    #create a column to indicate how many times this record was viewed#
    ###################################################################
    print"adding viewed total column..."
    #Group by the customer and the product
    total_viewed_group_by = model_data.groupby(['customer_ID','A','B','C','D','E','F','G']).size().reset_index()
    #relabel the last column as total views
    total_viewed_group_by.columns = ['customer_ID', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'total_viewed']

    #add total_viewed column to original dataset
    total_viewed_data = pd.merge(model_data[['customer_ID','shopping_pt','A','B','C','D','E','F','G']],total_viewed_group_by,on=['customer_ID','A','B','C','D','E','F','G'])[['customer_ID','shopping_pt','total_viewed']]
    print"converting states to floats..."
    ##convert state to floats to allow for categorical data processing
    state_dummies = pd.get_dummies(model_data['state'])
    state_data = model_data.join(state_dummies)[[u'customer_ID', u'shopping_pt', u'AL', u'AR',           u'CO', u'CT', u'DC', u'DE', u'FL', u'GA', u'IA', u'ID', u'IN', u'KS',           u'KY', u'MD', u'ME', u'MO', u'MS', u'MT', u'ND', u'NE', u'NH', u'NM',           u'NV', u'NY', u'OH', u'OK', u'OR', u'PA', u'RI', u'SD', u'TN', u'UT',           u'WA', u'WI', u'WV', u'WY']]
    print"converting car values to floats..."
    ##convert car values to floats to allow for categorical data processing
    car_value_dummies = pd.get_dummies(model_data['car_value'])
    car_value_data = model_data.join(car_value_dummies)[['customer_ID','shopping_pt',u'a', u'b',                           u'c',                 u'd',                 u'e',                           u'f',                 u'g',                 u'h',                           u'i']]

    original_model_data = model_data[['customer_ID','shopping_pt','day','location','group_size','homeowner','car_age',                                  'risk_factor','age_oldest','age_youngest','married_couple',                                  'C_previous','duration_previous', 'cost','hour','minute']][(model_data.record_type != 1)]
    print"merging all datasets..."
    all_new_data = pd.merge(car_value_data,                            pd.merge(state_data,                                     pd.merge(total_viewed_data,                                              pd.merge(is_last_data,is_final_data,                                                        on=['customer_ID','shopping_pt']),                                                           on=['customer_ID','shopping_pt']) ,                                                             on=['customer_ID','shopping_pt']),                                                                on=['customer_ID','shopping_pt'])
    print"creating final model..."
    final_model_data = pd.merge(original_model_data,all_new_data,on=['customer_ID','shopping_pt'],how='inner')

    X = np.asarray(final_model_data.ix[:, final_model_data.columns.difference(['customer_ID','shopping_pt','is_final'])])

    y = np.asarray(final_model_data.is_final)
    print"Done!"
    return X,y

def main():
    load_clean_data()

if __name__ == "__main__":
    main()


# In[ ]:

def grid_search(func, gridfunc=None, verbose=False, **kwargs):
    """
    Generic function for grid search

    :type func: callable
    :param func: function to perform grid search upon.

    :type gridfunc: callable
    :param gridfunc: a function for setting keyword's value based on the
    current parameters. Use this function to synthesize model name, ex.
    "mlp_lr0.001_nhidden100", or to set result directory, ex.
    "./result/test/mlp_hidden100".

    :type verbose: boolean
    :param verbose: to print out grid summaryy or not to

    :type kwargs: dict
    :param kwargs: dictionay of parameters for func. Keys should be valid input
    arguments of func, and their corresponding value could be a scalar or a list
    or values that would iterated over, ex, {'n_epochs:10, n_hidden:[100,150]'}

    Examples:

    >>> def test_func(a=1,b=2,c=3, prod=False):
            if prod is True:
                print("test_func: a*b*c=%d" % (a*b*c,))
            else:
                print("test_func: a=%d, b=%d, c=%d" % (a,b,c))

    >>> grid_search(test_func, verbose=True, **{'a':[1,2,3],'b':1})

    Use wrapper function to set input arguments:

    >>> def test_func_wrap(**kwargs):
            return test_func(b=1, prod=True, **kwargs)
    >>> grid_search(test_func_wrap, verbose=True, **{'a':[1,2,3]})

    Use gridfunc to change data directory

    >>> def gfunc(args):
            return {'dirname':'mlp_'+str(args['n_hidden'])}

    >>> grid_search(test_mlp, gridfunc=gfunc, **{'n_hidden':[100,200,300]})

    """

    # create parameters grid
    makelist = lambda x: x if hasattr(x, '__iter__') else (x,)
    kwargs = OrderedDict({k:makelist(v) for k,v in kwargs.items()})
    grid = product(*kwargs.values())
    n_grid = numpy.prod(map(len,(kwargs.values())))

    print('... starting grid search with %d combinations' % n_grid)
    for i, params in enumerate(grid):
        args = {k:v for k,v in zip(kwargs.keys(),params)}
        if gridfunc is not None:
            args.update(gridfunc(args))
        # print out parameters
        if verbose:
            print('=== Parameter set: %.4d/%.4d, %2.2f%% ===' %
                  (i+1,n_grid,100.*float(i+1)/float(n_grid)))
            for k,v in args.items():
                print('[%s]: %s' % (str(k),str(v)))
            print('... running function %s' % func.__name__)
        func(**args)
    print('... end of grid search\n')


