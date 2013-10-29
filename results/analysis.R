library(ggplot2)
library(Hmisc)

plot_one_metric <- function(df, metric) {
  g <- ggplot(combined, aes_string(x = 'feature', y = metric)) + 
    geom_boxplot() + 
    scale_x_discrete('Feature(s) Used', labels = c('averageSentenceLength' = 'Sent. Len.',
                                                   'averageWordLength' = 'Word Len.',
                                                   'get_avg_word_frequency_class' = 'Avg Word Freq Class',
                                                   'get_punctuation_percentage' = 'Punc. %',
                                                   'get_stopword_percentage' = 'StopWord %',
                                                   'all' = 'All')) +
    scale_y_continuous(capitalize(metric)) +
    ggtitle(capitalize(metric))
  
  print(g)  
}


set_feature_col <- function(df) {
	df$feature <- 'other'
	df[which(results$averageWordLength == 1), 'feature'] <- 'averageWordLength'
	df[which(results$averageSentenceLength == 1), 'feature'] <- 'averageSentenceLength'		
	df[which(results$get_avg_word_frequency_class == 1), 'feature'] <- 'get_avg_word_frequency_class'		
	df[which(results$get_punctuation_percentage == 1), 'feature'] <- 'get_punctuation_percentage'
	df[which(results$get_stopword_percentage == 1), 'feature'] <- 'get_stopword_percentage'		
	
	return(df)
}

set_accuracy_spec_cols <- function(df) {
  df$accuracy <- (df$true_pos + df$true_neg) / (df$true_pos + df$true_neg + df$false_pos + df$false_neg)
  df$precision <- df$true_pos / (df$true_pos + df$false_pos)
  # Note that recall can't really be definied for a non-plagarized text, since the number
  # of true pos = 0 and number of false neg = 0
  df$recall <- df$true_pos / (df$true_pos + df$false_neg)
  
  return(df)
}

results <- read.csv('results/initial_results/first_training_set_test.csv', head = T)
all_features <- read.csv('results/initial_results/full_set_first_training_test.csv', head = T)

# Add columns for accuracy, precision, recall
results <- set_accuracy_spec_cols(results)
all_features <- set_accuracy_spec_cols(all_features)

# Add a column showing which feature was used
results <- set_feature_col(results)
all_features$feature <- 'all'

# Combine the single-feature results with the using-all-features results
combined <- merge(results, all_features, all = T)

# Accuracy
plot_one_metric(combined, 'accuracy')
# precision
plot_one_metric(combined, 'precision')
# recall
plot_one_metric(combined, 'recall')

g <- ggplot(combined, aes(docname, accuracy, color = feature)) +
  geom_point()
print(g)

# word_length <- results[which(results$averageWordLength == 1), ]
# sent_length <- results[which(results$averageSentenceLength == 1), ]
# word_freq <- results[which(results$get_avg_word_frequency_class) == 1, ]
# punct_pct <- results[which(results$get_punctuation_percentage == 1), ]
# stopword_pct <- results[which(results$get_stopword_percentage == 1), ]

