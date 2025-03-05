import os
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import re 
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIRECTORY=f"{Path.cwd()}\data"

def read_data(file):
    colsname=["x1","y1","z1","x2","y2","z2","x3","y3","z3",]
    df = pd.read_csv(file, names=colsname, header=None)
    df["z3"] = df["z3"].str.replace(";","").astype("float64")
    df = df.astype("float64")
    return df

def normalize_data(df):
    # reduce datapoints
    df = df.loc[::2]
    # fix the g units 
    Sensor1_Resolution = 13
    Sensor1_Range = 16
    g_S1 = (2 * Sensor1_Range / 2 ** Sensor1_Resolution)

    # accelerations to g units by doing the calculations
    df.loc[:,'x1'] = g_S1 * df.loc[:,'x1']
    df.loc[:,'y1'] = g_S1 * df['y1']
    df.loc[:,'z1'] = g_S1 * df['z1']
    return df


def handle_file_fig_name(filename):
    tmpfile = os.path.basename(filename)  
    figfile = os.path.splitext(tmpfile)[0]
    return figfile

def create_data_sw(filename, df):
    x,y=df.shape
    in_wx=0
    namefig=handle_file_fig_name(filename)

    for wx in range(100,x+100,100):
        figure = f"{namefig}-{wx-100}-{wx}.jpg"
        ax=df.iloc[wx-100:wx, 0:3].plot(color=['black', 'black', 'black'])
        #ax.set_xlim(0, 12)  # Set X-axis scale (min, max)
        ax.set_ylim(-13,13)  # Set Y-axis scale (min, max)
        in_wx=wx
        #outputpath= f" - {DATA_DIRECTORY}\{figure}"
        ax.set_axis_off()
        ax.get_legend().remove()
        outputpath = Path(DATA_DIRECTORY) / figure
        print(outputpath)
        plt.savefig(outputpath, format='jpg', dpi=300)
        plt.close()

def create_truth(filename, df):
    x,y=df.shape
    in_wx=0
    namefig=handle_file_fig_name(filename)
    if re.match(r"^F\d{2}", namefig):
        for wx in range(100,x+100,100):
            flag=0
            figure = f"{namefig}-{wx-100}-{wx}.jpg"
            df2=df.iloc[wx-100:wx, 0:3]
            if ((df2['x1'].abs().max() > 2.5) or 
                (df2['y1'].abs().max() > 2.5) or 
                (df2['z1'].abs().max() > 2.5)):
                    flag=1
            with open("d:/source/FALLDETECTOR/train/the_truth.txt", "a") as f:
                #f.write(f"{figure}: {flag}")
                f.write(f"{figure}:{flag}\n")
            f.close()
          
            # print(f"{figure}: {flag}")
    else:
        for wx in range(100,x+100,100):
            flag=0
            figure = f"{namefig}-{wx-100}-{wx}.jpg"
            #print(f"{figure}: {flag}")
            with open("d:/source/FALLDETECTOR/train/the_truth.txt", "a") as f:
                #f.write(f"{figure}: {flag}")
                f.write(f"{figure}:{flag}\n")
            f.close()

# Define base data path
data_path = Path("D:/source/Falldetector/SisFall_dataset")

# Regular expression to match folders "SA01-SA15" and "SE01-SE15"
pattern = re.compile(r"^(SA|SE)(0[1-9]|1[0-5])$")

# Get only matching subdirectories
filtered_dirs = [d for d in data_path.iterdir() if d.is_dir() and pattern.match(d.name)]

# Print total matching folders
print(f"Total matching folders: {len(filtered_dirs)}\n")

# Iterate through matching directories
for folder in filtered_dirs:
    #print(f"📂 {folder}")  # Print folder path

    # Get all .txt files in the folder
    txt_files = [file for file in folder.iterdir() if file.suffix == ".txt"]

    # Print .txt files
    for txt_file in txt_files:
        print(f"{txt_file.resolve()}")
        df=read_data(txt_file.resolve())
        df2=normalize_data(df)
        create_data_sw(txt_file.resolve(),df2)
        create_truth(txt_file.resolve(),df2)
    
#df = pd.read_csv("D:/source/Falldetector/SisFall_dataset/SA01/D01_SA01_R01.txt")
