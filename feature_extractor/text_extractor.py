import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract(model, t):
    res = model.forward(**t)
    return res.pooler_output.cpu().numpy()

def extract_sbert(model, t):
    return model.encode(t, show_progress_bar=False)


def get_text_feature(videoSentences):
    videoText = {}
    video_sentence = videoSentences

    model = SentenceTransformer("paraphrase-distilroberta-base-v1")
    model.to(device)
    for k in tqdm(video_sentence):
        videoText[k] = extract_sbert(model, video_sentence[k])

    return videoText
