# NeST

This is the code for the paper `[Neighborhood-regularized Self-training for Learning with Few Labels]()' (In Proceedings of AAAI 2023).

# Requirements
```
python 3.7
transformers==4.2.0
pytorch==1.8.0
tqdm
scikit-learn
faiss-cpu==1.6.4
```

# Datasets
## Datasets

The datasets used in this study can be find at the following link

|   Dataset   | Task  | Number of Classes | Number of Train/Test |
|---------------- | -------------- |-------------- | -------------- |
| [Elec](http://riejohnson.com/cnn_data.html) |    Sentiment    |      2      |  25K / 25K   |
| [AG News](https://huggingface.co/datasets/ag_news)    |    News Topic          |     2   |  120K / 7.6K  |
| [NYT](https://github.com/yumeng5/CatE/tree/master/datasets/nyt)  |  News Topic   |    4     |     30K / 3.0K    |
| [Chemprot](https://github.com/yueyu1030/COSINE/tree/main/data/chemprot)     |     Chemical Relation      |    10    |    12K / 1.6K     |

## Input Format
"_id" stands for the class id, and "text" is the content of the document.
```
    {"_id": 0, "text": "Congo Official: Rwanda Troops Attacking (AP) AP - A senior Congolese official said Tuesday his nation had been invaded by neighboring Rwanda, and U.N. officials said they were investigating claims of Rwandan forces clashing with militias in the east."}
    {"_id": 1, "text": "Stadler Leads First Tee Open (AP) AP - Craig Stadler moved into position for his second straight victory Saturday, shooting a 9-under 63 to take a one-stroke lead over Jay Haas after the second round of the inaugural First Tee Open."}
    {"_id": 2, "text": "Intel Shares Edge Lower After Downgrade  NEW YORK (Reuters) - Intel Corp shares slipped on  Tuesday after Credit Suisse First Boston downgraded the stock,  forecasting that the computer chip maker will have difficulty  outperforming the overall semiconductor sector next year."}
    {"_id": 3, "text": "Debating the Dinosaur Extinction At least 50 percent of the world's species, including the dinosaurs, went extinct 65 million years ago. While most scientists now blame this catastrophe on a large meteorite impact, others wonder if there is more to the story."}
    ...
}
```

## Training
Please use the commands in `commands` folder for experiments.
Take AG News dataset as an example, `run_agnews.sh` is used for running the experiment for self-training.



# Hyperparameter Tuning
Some Key Hyperparameters are listed as follows
- `k`: The number of nearest neighbors used in KNN.
- `learning_rate`: The learning rate for initialzation.
- `learning_rate_st`: The learning rate for self-training.
- `self_training_update_period`:  The update period of self-training. 
- `self_training_weight`: The weight to balance labeled data and unlabeled data during self-training.
- `num_unlabeled`:  The number of unlabeled data in the beginning.
- `num_unlabeled_add`: The number of added unlabeled data in each self-training round. 


# Citation 

Please kindly cite the following paper if you are using our datasets/codebase. Thanks!

```
@inproceedings{xu2023neighborhood,
    title = "Neighborhood-regularized Self-training for Learning with Few Labels",
    author = "Ran Xu and Yue Yu and Hejie Cui and Xuan Kan and Yanqiao Zhu and Joyce C. Ho and Chao Zhang and Carl Yang",
    booktitle = "Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence",
    year = "2023",
}
```