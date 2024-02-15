from extractions import *
from loads import *
from transforms import *

data = read_from_csv("./data/orders.csv")
#print(data)
#--------------------------------------------------
data = drop_nans_records(data)
#print(data)
#--------------------------------------------------
data = fill_noisy_data(data)
#print(data)
#--------------------------------------------------
data = fill_NaN(data)
#print(data)
#-------------------------------------------------
data = k_bins_discretizer(data,["Quantity"])  
#print(data)
#-------------------------------------------------
data= label_encoding(data, ['Color'])
#print(data)
drop_columns(data, ["CustomerName"])

#------------------------------------------------
#check_outliear_column_by_plotly(data,['PurchaseAmount','Weight'])
data = remove_purchaseAmount_outliears(data, 100000, 3000000)
#print(data)

#------------------------------------------------
data = standard_scaler(data,["PurchaseDate", "Product","Quantity","PurchaseAmount","Color","IsAvailable" ,"Weight"])
#print(data)
load(data, "./targetFile.csv")

