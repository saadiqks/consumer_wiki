import cohere
from cohere import ClassifyExample
import os
import polars as pl
import itertools as it
from tqdm import tqdm

def label_titles_test():
    df = pl.read_csv("unlabelled_data.csv")
    small_df = df.head(n=300).sample(fraction=1, shuffle=True)

    train_df = small_df.head(n=290)
    test_df = small_df.tail(n=10)

    examples = []
    inputs = []
    preds = []

    for row in train_df.select(["video_title", "needs_wiki_article"]).rows():
        elem = ClassifyExample(text=row[0], label=row[1])
        examples.append(elem)

    api_key = os.getenv("COHERE_API_KEY")
    co = cohere.Client(api_key=api_key)

    test = [row[0] for row in test_df.select(["video_title", "needs_wiki_article"]).rows()]
    actual = [row[1] for row in test_df.select(["video_title", "needs_wiki_article"]).rows()]

    response = co.classify(
        inputs=test,
        examples=examples,
    )

    for elem in response.classifications:
       # print("Input: ", elem.input)
       # print("Label: ", elem.prediction)

       if elem.input:
           inputs.append(elem.input)
           preds.append(elem.prediction)

    print("Accuracy: ", (sum(1 for i in range(10) if preds[i] == actual[i]) / 10) * 100)


def label_titles():
    df = pl.read_csv("unlabelled_data.csv")
    train_df = df.head(n=300)
    unlabelled_df = df.tail(n=3421)

    examples = []
    inputs = []
    preds = []

    for row in train_df.select(["video_title", "needs_wiki_article"]).rows():
        elem = ClassifyExample(text=row[0], label=row[1])
        examples.append(elem)

    api_key = os.getenv("COHERE_API_KEY")
    co = cohere.Client(api_key=api_key)
    video_titles = [row[0] for row in unlabelled_df.select("video_title").rows()]

    for batch in tqdm(it.batched(video_titles, n=96)):
        response = co.classify(
            inputs=batch,
            examples=examples,
        )

        for elem in response.classifications:
            if elem.input:
                inputs.append(elem.input)
                preds.append(elem.prediction)

    res_df = pl.DataFrame({"video_title": inputs, "needs_wiki_article": preds})
    res_df.write_csv("labelled_data.csv")


def merge_all_data():
    labelled_df = pl.read_csv("labelled_data.csv")
    unlabelled_df = pl.read_csv("unlabelled_data.csv")
    video_titles = []
    needs_wiki_article_list = []

    for row in unlabelled_df.select(["video_title", "needs_wiki_article"]).head(n=300).rows() + labelled_df.select(["video_title", "needs_wiki_article"]).rows():
        video_titles.append(row[0])
        needs_wiki_article_list.append(row[1])

    final_df = pl.DataFrame({
        "video_title": video_titles,
        "needs_wiki_article": needs_wiki_article_list,
        "wiki_url": unlabelled_df["wiki_url"],
        "video_url": unlabelled_df["video_url"]
    })

    final_df.write_csv("final_data.csv")


def main():
    merge_all_data()


if __name__ == "__main__":
    main()
