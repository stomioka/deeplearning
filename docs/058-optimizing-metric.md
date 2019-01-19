# Satisficing and Optimizing metric

It's not always easy to combine all the things you care about into a single row number evaluation metric. Sometimes it is useful to set up satisficing as well as optimizing matrix.

## Example 1
Suppose you care about both the classification accuracy and the running time of a learning algorithm. You need to choose from these three classifiers:

![](images/058-optimizing-metric-07bc29fb.png)

It seems unnatural to derive a single metric by putting accuracy and running time into a single formula, such as:
cost = Accuracy - 0.5*RunningTime

But maybe it seems a bit artificial to combine accuracy and running time using a formula like this, like a linear weighted sum of these two things.

Here’s what you can do instead:
1. Define what is an “acceptable” running time. Lets say anything that runs in 100ms is acceptable. Then,
2. Maximize accuracy, subject to your classifier meeting the running time criteria. Here, running time is a **“satisficing metric”** —your classifier just has to be “good enough” on this metric, in the sense that it should take at most 100ms. Accuracy is the **“optimizing metric.”**. Classification B becomes the classification to be used.

![](images/058-optimizing-metric-f555e138.png)

More generally, if you have

**N metrics: use 1 metric for optimizing, use N-1 for satisficing metrics**

## Example 2

As a final example, suppose you are building a hardware device that uses a microphone to listen for the user saying a particular _“wakeword”_ or _"trogger words"_, that then causes the system to wake up. Examples include Amazon Echo listening for _“Alexa”_; Apple Siri listening for _“Hey Siri”_; Android listening for _“Okay Google”_; and Baidu apps listening for _“你好百度”_ You care about both the **false positive rate**—the frequency with which the system wakes up even when no one said the _wakeword_—as well as the **accuracy rate**—how often it wake up when someone says the _wakeword_. One reasonable goal for the performance of this system is to **maximize the accuracy rate (optimizing metric)**, subject to there being **no more than one false positive every 24 hours of operation (satisficing metric)**.
Once your team is aligned on the evaluation metric to optimize, they will be able to make faster progress.
