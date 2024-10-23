Fake News Detection

This working model of machine learning can detect the accuracy of the news by using the past data presenet with us

Let’s say we have two news articles:

1. Real News:

Headline: "NASA Confirms Water on the Moon"

Body: "NASA has confirmed the presence of water on the moon through recent satellite data. Scientists believe this discovery could help future lunar missions. The water is located in shadowed craters, and it could potentially be used for drinking or making fuel."



2. Fake News:

Headline: "NASA Hides Evidence of Alien Life on the Moon"

Body: "An anonymous source from NASA leaked documents proving that the space agency has been hiding evidence of alien life found on the moon. This conspiracy has been going on for decades, and many believe the government is covering it up."




How the Model Detects Fake News:

Step 1: Preprocessing

Both articles are processed and transformed into a numerical format using TF-IDF. During this step, words like “the,” “on,” and “has” (common across both articles) get lower scores because they don’t provide unique information. However, certain words get higher TF-IDF scores:

Real News (high TF-IDF scores): "NASA," "water," "moon," "discovery," "scientists," "missions"

Fake News (high TF-IDF scores): "aliens," "conspiracy," "hides," "government," "leaked," "anonymous"


Words like “conspiracy,” “aliens,” and “hides” are relatively rare and specific to fake news articles in the training dataset, so their TF-IDF scores are high. Similarly, words like “water,” “discovery,” and “scientists” might frequently appear in legitimate science news, so they get high scores in real news articles.

Step 2: Model Training

The Passive-Aggressive Classifier has been trained on a dataset containing both real and fake news articles. During training:

The classifier learns to associate certain words or combinations of words with real news (e.g., "NASA," "discovery," "confirmed").

It also learns that certain words or phrases are more common in fake news (e.g., "conspiracy," "aliens," "leaked," "hides").


The model doesn’t just look at individual words but how often they appear in fake or real articles across the entire dataset.

Step 3: Making Predictions

When the model encounters the new fake news article, it uses the words that are most important (those with high TF-IDF scores) to make a decision.

In the fake news article, words like “conspiracy,” “aliens,” “hides,” and “leaked” stand out because they have high TF-IDF scores and the model has learned that these words are often associated with fake news.

The real news article contains words like “discovery,” “water,” and “scientists,” which the model has learned to associate with real news.


Step 4: Decision Making (Passive-Aggressive Classifier)

If the model has previously seen similar patterns (high TF-IDF scores for words like "conspiracy" and "aliens") in fake news articles, it will predict that the second article is likely fake.

On the other hand, if the model recognizes patterns typical of real news articles (like "discovery" or "scientists"), it will predict the first article as real.


Since the model has been trained to detect fake news, it can detect the second article as fake based on the pattern of words it uses, even though it hasn’t seen this exact article before.

Step 5: Learning (Aggressive Update if Needed)

If the model makes a wrong prediction, it adjusts its weights. For example:

If it wrongly classifies the fake article as real, the Passive-Aggressive Classifier would aggressively update its weights to give more importance to words like "conspiracy" and "aliens" so that it can correctly classify similar articles in the future.


Why Does the Model Work?

TF-IDF helps the model focus on the important words that differentiate real from fake news, rather than just looking at all the words equally.

The Passive-Aggressive Classifier ensures that the model learns quickly from mistakes and gets better over time by adjusting its decision-making process based on the errors it makes.


Summary:

The fake news article contains words that the model has learned are commonly associated with fake news, like “conspiracy” or “aliens.” When these words get high TF-IDF scores, the model is more likely to classify the article as fake. The model’s ability to recognize these patterns (through TF-IDF) and learn from its mistakes (through the Passive-Aggressive Classifier) is what makes it effective in detecting fake news.
![image](https://github.com/user-attachments/assets/fa6f15eb-831c-4eab-8ded-c470e9865152)
