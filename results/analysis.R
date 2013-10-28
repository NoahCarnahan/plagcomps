library(plyr)

one_summary <- function(df, feature) {
	only_feature <- df[which(df[, feature] == 1), ]
	print(head(only_feature))
	
	hist(only_feature$accuracy)
}


results <- read.csv('results/first_training_set_test.csv', head = T)

results$accuracy <- (results$true_pos + results$true_neg) / (results$true_pos + results$true_neg + results$false_pos + results$false_neg)
summary(results$accuracy)

one_summary(results, 'averageWordLength')


word_length <- results[which(results$averageWordLength == 1), ]
summary(word_length$accuracy)

sent_length <- results[which(results$averageSentenceLength == 1), ]

word_freq <- results[which(results$get_avg_word_frequency_class) == 1, ]

punct_pct <- results[which(results$get_punctuation_percentage == 1), ]

stopword_pct <- results[which(results$get_stopword_percentage == 1), ]


names(results)