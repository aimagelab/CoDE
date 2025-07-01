from visualization import get_features
import torch.distributed as dist
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import os
import joblib
from pydantic import BaseModel
from typing import List
import numpy as np
import json
import torch.nn as nn
import wandb
from sklearn import svm

from collections import OrderedDict

__all__ = ['classifier_training', 'classifier_eval', 'TrainingResults', 'Model', 'Models']


class TrainingResults(BaseModel):
    maccuracy: float
    real_accuracy: float
    fake_accuracy: float
    fake_accuracy_0: float = 0
    fake_accuracy_1: float = 0
    fake_accuracy_2: float = 0
    fake_accuracy_3: float = 0


def score_function(features, y_true, model, indexes=None, svm=False):
    if indexes is None:
        indexes = np.ones(len(y_true), dtype=bool)
    if svm:
        predictions = model.predict(features[indexes])
        scores = np.where(predictions == -1, 1, 0)
        return np.sum(scores == y_true[indexes]) / len(y_true[indexes])
    return model.score(features[indexes], y_true[indexes])


class Model(BaseModel):
    model_name: str
    model_path: str


class Models(BaseModel):
    models: List[Model]


def quantitative_evaluation_linear_training(features: np.ndarray, labels: np.ndarray, labels_train: np.array, model,
                                            svm=False):
    index_real = np.where(labels == 0, True, False)
    index_fake = np.where(labels != 0, True, False)
    index_gen_0 = np.where(labels == 1, True, False)
    index_gen_1 = np.where(labels == 2, True, False)
    index_gen_2 = np.where(labels == 3, True, False)
    index_gen_3 = np.where(labels == 4, True, False)
    predictions = model.predict(features)

    mean_accuracy = score_function(features, labels_train, model, svm=svm)
    # real_accuracy = model.score(features[index_real], labels_train[index_real])
    real_accuracy = score_function(features, labels_train, model, indexes=index_real, svm=svm)
    # fake_accuracy = model.score(features[index_fake], labels_train[index_fake])
    fake_accuracy = score_function(features, labels_train, model, indexes=index_fake, svm=svm)
    if np.any(labels == 1):
        # fake_accuracy_0 = model.score(features[index_gen_0], labels_train[index_gen_0])
        fake_accuracy_0 = score_function(features, labels_train, model, indexes=index_gen_0, svm=svm)
    else:
        fake_accuracy_0 = 0
    if np.any(labels == 2):
        # fake_accuracy_1 = model.score(features[index_gen_1], labels_train[index_gen_1])
        fake_accuracy_1 = score_function(features, labels_train, model, indexes=index_gen_1, svm=svm)
    else:
        fake_accuracy_1 = 0
    if np.any(labels == 3):
        # fake_accuracy_2 = model.score(features[index_gen_2], labels_train[index_gen_2])
        fake_accuracy_2 = score_function(features, labels_train, model, indexes=index_gen_2, svm=svm)
    else:
        fake_accuracy_2 = 0
    if np.any(labels == 4):
        # fake_accuracy_3 = model.score(features[index_gen_3], labels_train[index_gen_3])
        fake_accuracy_3 = score_function(features, labels_train, model, indexes=index_gen_3, svm=svm)
    else:
        fake_accuracy_3 = 0
    return TrainingResults(maccuracy=mean_accuracy, real_accuracy=real_accuracy, fake_accuracy=fake_accuracy,
                        fake_accuracy_0=fake_accuracy_0,
                           fake_accuracy_1=fake_accuracy_1,
                           fake_accuracy_2=fake_accuracy_2, fake_accuracy_3=fake_accuracy_3)


def classifier_training(dataloader, model: nn.Module, eval_model: Models, output_dir, epoch, args, distributed=True):
    # ----------------- Training -----------------
    classifiers = {}
    test_features, test_labels = get_features(dataloader, model, args)
    if distributed:
        print(f"Distributed, gathering features and labels {dist.get_rank()}")
        print(f"Rank {dist.get_rank()}: Distributed, before gather test_features shape {test_features.shape}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        dist.barrier()
        gather_list_features = [torch.zeros_like(test_features) for _ in range(world_size)]
        dist.all_gather(gather_list_features, test_features)
        gather_list_labels = [torch.zeros_like(test_labels) for _ in range(world_size)]
        dist.all_gather(gather_list_labels, test_labels)
    else:
        rank = 0
        gather_list_features = [test_features]
        gather_list_labels = [test_labels]
    dirs = []
    if rank == 0:
        concatenated_features = torch.cat(gather_list_features, dim=0)
        concatenated_labels = torch.cat(gather_list_labels, dim=0)
        print(f"Rank {dist.get_rank()}: Distributed, after gather test_features shape {concatenated_features.shape}")
        X = concatenated_features.cpu().numpy()
        Y = concatenated_labels.cpu().numpy()  # 0 = real, 1 = fake_0, 2 = fake_1, 3 = fake_2, 4 = fake_3
        Y_train = np.where(Y == 0, 0, 1)  # 0 = real, 1 = fake
        if args.auc_for_rebuttal:
            classifier = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric="cosine", n_jobs=-1)
            index_real = np.where(Y_train == 0, True, False)
            classifier.fit(X[index_real], Y[index_real])
            file_name = os.path.join(output_dir, "knn_oneclass0.sav")
            joblib.dump(classifier, file_name)
            classifier = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric="cosine", n_jobs=-1)
            index_fake = np.where(Y_train == 1, True, False)
            classifier.fit(X[index_fake], Y[index_fake])
            file_name = os.path.join(output_dir, "knn_oneclass1.sav")
            joblib.dump(classifier, file_name)
            classifier = joblib.load(args.classifier_checkpoint)
            values = classifier.decision_function(X[index_real])
            torch.save(values, os.path.join(output_dir, "decision_function_trainingsplit.pth"))
        for class_model in eval_model.models:
            dir = os.path.join(output_dir, f"{class_model.model_name}_{epoch}_fixed")
            os.makedirs(dir, exist_ok=True)
            dirs.append(dir)
            if args.multiple_evaluations:
                for generator in range(1, 5):
                    if class_model.model_name == 'linear':
                        classifier = LogisticRegression(random_state=args.seed, C=0.316,
                                                        max_iter=args.epochs_classifier,
                                                        verbose=1,
                                                        class_weight='balanced')
                    elif class_model.model_name == 'knn':
                        classifier = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric="cosine", n_jobs=-1)
                    elif class_model.model_name == 'svm':
                        continue
                    else:
                        raise Exception(f"Model {class_model.model_name} not implemented")
                    index_gen = np.where(Y == 0, True, False) + np.where(Y == generator, True, False)
                    classifier.fit(X[index_gen], Y_train[index_gen])

                    file_name = f"{class_model.model_name}_generator-{generator - 1}_classifier_epoch-{epoch}.sav"
                    # create a directory if it does not exist
                    file_name = os.path.join(dir, file_name)
                    joblib.dump(classifier, file_name)
                    # ----------------- Evaluation on training samples -----------------
                    train_results = quantitative_evaluation_linear_training(X, Y, Y_train, classifier)
                    with open(os.path.join(dir, f"{class_model.model_name}_train_epoch-{epoch}.json"), "a") as f:
                        json.dump({f"gen_{generator - 1}": train_results.model_dump()}, f)
                    classifiers[
                        f"{class_model.model_name}_generator-{generator - 1}_classifier_epoch-{epoch}"] = classifier
            if class_model.model_name == 'linear':
                classifier = LogisticRegression(random_state=args.seed, C=0.316, max_iter=args.epochs_classifier,
                                                verbose=1,
                                                class_weight='balanced')
            elif class_model.model_name == 'knn':
                classifier = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric="cosine", n_jobs=-1)
            elif class_model.model_name == 'svm':
                classifier = svm.OneClassSVM(nu=0.1, kernel="poly", gamma='auto')
                index_real = np.where(Y == 0, True, False)
                buffer_x = X
                buffer_y = Y_train
                X = X[index_real]
                Y_train = Y_train[index_real]

            else:
                raise Exception(f"Model {class_model.model_name} not implemented")

            classifier.fit(X, Y_train)
            if class_model.model_name == 'svm':
                X = buffer_x
                Y_train = buffer_y
            file_name = f"{class_model.model_name}_tot_classifier_epoch-{epoch}.sav"
            # create a directory if it does not exist
            file_name = os.path.join(dir, file_name)
            joblib.dump(classifier, file_name)
            # ----------------- Evaluation on training samples -----------------
            train_results = quantitative_evaluation_linear_training(X, Y, Y_train, classifier,
                                                                    svm=class_model.model_name == 'svm')
            with open(os.path.join(dir, f"{class_model.model_name}_train_epoch-{epoch}.json"), "a") as f:
                json.dump({f"tot": train_results.model_dump()}, f)
            classifiers[f"{class_model.model_name}_tot_classifier_epoch-{epoch}"] = classifier
    return classifiers, dirs


def classifier_eval(dataloader, classifiers: dict, model: torch.nn.Module, output_dir,
                    epoch: int, args, distributed=True, transf=False, dirs=None):
    # ----------------- Training -----------------
    test_features, test_labels = get_features(dataloader, model, args)
    if distributed:
        print(f"Distributed, gathering features and labels {dist.get_rank()}")
        print(f"Rank {dist.get_rank()}: Distributed, before gather test_features shape {test_features.shape}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        dist.barrier()
        gather_list_features = [torch.zeros_like(test_features) for _ in range(world_size)]
        dist.all_gather(gather_list_features, test_features)
        gather_list_labels = [torch.zeros_like(test_labels) for _ in range(world_size)]
        dist.all_gather(gather_list_labels, test_labels)
    else:
        rank = 0
        gather_list_features = [test_features]
        gather_list_labels = [test_labels]

    if rank == 0:
        concatenated_features = torch.cat(gather_list_features, dim=0)
        concatenated_labels = torch.cat(gather_list_labels, dim=0)
        print(f"Rank {dist.get_rank()}: Distributed, after gather test_features shape {concatenated_features.shape}")
        X = concatenated_features.cpu().numpy()
        Y = concatenated_labels.cpu().numpy()
        Y_train = np.where(Y == 0, 0, 1)
        linear_results = {}
        knn_results = {}
        svm_results = {}
        dir_linear = ""
        dir_knn = ""
        dir_svm = ""
        for output_dir in dirs:
            if 'linear' in output_dir:
                dir_linear = output_dir
            if 'svm' in output_dir:
                dir_svm = output_dir
            if 'knn' in output_dir:
                dir_knn = output_dir
        for classifier_folder in classifiers.keys():
            linear = True if 'linear' in classifier_folder else False
            svm = True if 'svm' in classifier_folder else False
            knn = True if 'knn' in classifier_folder else False

            eval_results = quantitative_evaluation_linear_training(X, Y, Y_train, classifiers[classifier_folder],svm=svm)
            if args.log_wandb and "tot_classifier" in classifier_folder:
                train_results_dict = dict(eval_results)
                rowd = OrderedDict()
                # TODO: check if it is logging only the last one (thus the all gen?)
                rowd.update(
                    [(f'eval_{"linear" if "linear" in classifier_folder else ""}{"svm" if "svm" in classifier_folder else ""}{"knn" if "knn" in classifier_folder else ""}_tot_classifier{"_transf" if transf else ""}' + k, v) for k, v in
                     train_results_dict.items()])
                wandb.log(rowd, commit=False)
            if linear:
                linear_results[classifier_folder] = eval_results.model_dump()
            if knn:
                knn_results[classifier_folder] = eval_results.model_dump()
            if svm:
                svm_results[classifier_folder] = eval_results.model_dump()
        if dir_linear != "":
            with open(os.path.join(dir_linear, f"linear_eval{'_transf' if transf else ''}.json"), "w") as f:
                json.dump(linear_results, f)
        if dir_knn != "":
            with open(os.path.join(dir_knn, f"knn_eval{'_transf' if transf else ''}.json"), "w") as f:
                json.dump(knn_results, f)
        if dir_svm != "":
            with open(os.path.join(dir_svm, f"svm_eval{'_transf' if transf else ''}.json"), "w") as f:
                json.dump(svm_results, f)
    else:
        # dummy values for not master ranks
        X = np.ones(1)
        Y = np.ones(1)
        linear_results = {}
    print(f"Rank {dist.get_rank()}: Finished evaluation, waiting for other ranks to finish")
    dist.barrier()
    # return a numpy array for avoiding other ranks to extract features again
    return X, Y, linear_results

