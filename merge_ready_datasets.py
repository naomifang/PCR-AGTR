"""Merge prepared per-dataset CSV files for shared feature extraction."""
from pathlib import Path
import argparse
import pandas as pd

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--ready-root",required=True)
    parser.add_argument("--datasets",nargs="+",default=["DRH","Kinase","SNAP"])
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    root=Path(args.ready_root)
    frames=[]
    for dataset in args.datasets:
        path=root/dataset/f"{dataset}_ready.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    merged=pd.concat(frames,ignore_index=True).drop_duplicates().reset_index(drop=True)
    merged.to_csv(output,index=False)
    print({"output":str(output),"rows":len(merged),"datasets":args.datasets})

if __name__=="__main__":
    main()
