import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer,LabelEncoder
def drop_nans_records(data):
    return data.dropna(how="all", axis=0)
#-------------------------------------------------
def fill_NaN(data):
    data=data.fillna(value={
        "CustomerName" : data.CustomerName.mode()[0],
        "Product": data.Product.mode()[0],
        "Quantity": data.Quantity.mean(),
        "PurchaseAmount": data.PurchaseAmount.mean(),
        "Color": data.Color.mode()[0],
        "IsAvailable": data.IsAvailable.mode()[0],
        "Weight": data.Weight.mean(),
    })
    data = data.bfill()
    return data
#-------------------------------------------------
def fill_noisy_data(data):
    data["Weight"] = pd.to_numeric(data["Weight"], errors="coerce")
    data["PurchaseAmount"] = pd.to_numeric(data["PurchaseAmount"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["PurchaseDate"] = pd.to_datetime(data["PurchaseDate"])
    data.loc[data.CustomerName.astype("str").str.isnumeric(),"CustomerName"]
    data.loc[data.Color.astype("str").str.isnumeric(),"Color"]
    data.loc[data.Color.astype('bool'), 'Color']
    data.loc[data.Product.astype("str").str.isnumeric(),"Product"]
    return data

#-------------------------------------------------
def k_bins_discretizer(data, columns):
    dis = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
    for col in columns:
        data[col] = dis.fit_transform(data[[col]])
    return data
#-------------------------------------------------
def label_encoding(data, columns):
    le = LabelEncoder()
    for col in columns:
        data[col] = le.fit_transform(data[col])
    return data
#-------------------------------------------------
import plotly.express as px
def check_outliear_column_by_plotly(data, columns):
    fig = px.box(data, y=columns)
    fig.show()

def remove_purchaseAmount_outliears(data, min_w, max_w):
    df=pd.DataFrame(data)
    data=df[(df["PurchaseAmount"] >= min_w) & (df["PurchaseAmount"] <= max_w)]
    return 
#-------------------------------------------------
def drop_columns(data, columns):
    for col in columns:
        data.drop(col, axis=1, inplace=True)
    return data

#-------------------------------------------------
from sklearn.preprocessing import StandardScaler
def standard_scaler(data, columns):
    scaler = StandardScaler()
    data=scaler.fit_transform(data)
    data = pd.DataFrame(data) 
    data.columns = columns
    return data