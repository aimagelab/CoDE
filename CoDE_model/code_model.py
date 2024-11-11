import transformers
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
import torch.nn as nn
import joblib
from torchvision import transforms

"""
linear - knn --> 0 Real - 1 Fake
svm --> -1 Real - 1 Fake
"""


class VITContrastiveHF(nn.Module):
    """
    This class is a wrapper for the CoDE model. It is used to load the model and the classifier
    """

    def __init__(self, classificator_type):
        """
        Constructor of the class
        :param repo_name: the name of the repository
        :param classificator_type: the type of the classifier
        """
        super(VITContrastiveHF, self).__init__()
        self.model = transformers.AutoModel.from_pretrained("aimagelab/CoDE")
        self.model.pooler = nn.Identity()
        self.classificator_type = classificator_type
        self.processor = transformers.AutoProcessor.from_pretrained("aimagelab/CoDE")
        self.processor.do_resize = False
        # define the correct classifier /// consider to use the `cache_dir`` parameter
        if classificator_type == "svm":
            file_path = hf_hub_download(
                repo_id="aimagelab/CoDE",
                filename="sklearn/ocsvm_kernel_poly_gamma_auto_nu_0_1_crop.joblib",
            )
            self.classifier = joblib.load(file_path)

        elif classificator_type == "linear":
            file_path = hf_hub_download(
                repo_id="aimagelab/CoDE",
                filename="sklearn/linear_tot_classifier_epoch-32.sav",
            )
            self.classifier = joblib.load(file_path)

        elif classificator_type == "knn":
            file_path = hf_hub_download(
                repo_id="aimagelab/CoDE",
                filename="sklearn/knn_tot_classifier_epoch-32.sav",
            )
            self.classifier = joblib.load(file_path)

        else:
            raise ValueError("Selected an invalid classifier")

    def forward(self, x, return_feature=False):
        features = self.model(x)
        if return_feature:
            return features
        features = features.last_hidden_state[:, 0, :].cpu().detach().numpy()
        predictions = self.classifier.predict(features)
        return torch.from_numpy(predictions)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # HF inference code
    classificator_type = "linear"
    model = VITContrastiveHF(
        repo_name="aimagelab/CoDE", classificator_type=classificator_type
    )

    transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model.eval()
    model.model.to(device)
    y_pred = []
    img = Image.open("206496010652.png").convert("RGB")

    with torch.no_grad():
        # in_tens = model.processor(img, return_tensors='pt')['pixel_values']
        in_tens = transform(img).unsqueeze(0)

        in_tens = in_tens.to(device)
        y_pred.extend(model(in_tens).flatten().tolist())

    # check the correct label of the predict image
    for el in y_pred:
        if el == 1:
            print("Fake")
        elif el == 0:
            print("Real")
        elif el == -1:
            print("Real")
        else:
            print("Error")
