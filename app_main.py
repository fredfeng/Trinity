import uuid
import os
import json
import psutil
import time

import app_synthesize

from tyrell.logger import get_logger
logger = get_logger('tyrell')
logger.setLevel('DEBUG')

request_config = {
	"title": "How can I spread repeated measures of multiple variables into wide format?",
	"question": 
"""I'm trying to take columns that are in long format and spread them to wide format as shown below. I'd like to use tidyr to solve this with the data manipulation tools I'm investing in but to make this answer more general please provide other solutions.

Here's what I have:

`library(dplyr); library(tidyr)

set.seed(10)
dat <- data_frame(
    Person = rep(c("greg", "sally", "sue"), each=2),
    Time = rep(c("Pre", "Post"), 3),
    Score1 = round(rnorm(6, mean = 80, sd=4), 0),
    Score2 = round(jitter(Score1, 15), 0),
    Score3 = 5 + (Score1 + Score2)/2
)

##   Person Time Score1 Score2 Score3
## 1   greg  Pre     80     78   84.0
## 2   greg Post     79     80   84.5
## 3  sally  Pre     75     74   79.5
## 4  sally Post     78     78   83.0
## 5    sue  Pre     81     78   84.5
## 6    sue Post     82     81   86.5`
Desired wide format:

`  Person Pre.Score1 Pre.Score2 Pre.Score3  Post.Score1 Post.Score2 Post.Score3
1   greg         80         78       84.0           79          80        84.5
2  sally         75         74       79.5           78          78        83.0
3    sue         81         78       84.5           82          81        86.5`
I can do it by doing something like this for each score:

`spread(dat %>% select(Person, Time, Score1), Time, Score1) %>% 
    rename(Score1_Pre = Pre, Score1_Post = Post)`
And then using `_join` but that seems verbose and like there's got to be a better way.

Related questions:
tidyr wide to long with two repeated measures
Is it possible to use spread on multiple columns in tidyr similar to dcast?""",
	"size": 3,
	"input0": [
		["Person","Time","Score1","Score2"],
		["greg","Pre",88,84],
		["greg","Post",78,82],
		["sally","Pre",76,72],
		["sally","Post",78,79],
	],
	"input1": None,
	"output": [
		["Person","Post_Score1","Post_Score2","Pre_Score1","Pre_Score2"],
		["greg",78,82,88,84],
		["sally",78,79,76,72],
	],
	"spec": "morpheus",
	"use_nl": True,
}
resp_start = time.time()
resp = app_synthesize.synthesize(request_config)
resp_end = time.time()
resp["time"] = resp_end - resp_start
