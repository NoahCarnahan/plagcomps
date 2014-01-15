library(ggplot2)
library(Hmisc)

# Boxplot of <metric> broken down by whether the passage was plagiarized
plot_one_metric <- function(df, metric, doc_name='foo') {
  g <- ggplot(df) +
    geom_boxplot(aes_string(x = 'contains_plag', y = metric)) +
    scale_x_discrete('Contains Plag?', labels = c('No', 'Yes')) +
    ggtitle(doc_name)
  print(g)  
}


# Scatterplot of feat1 on x-axis, feat2 on y-axis and points 
# colored by whether or not the passage was plagiarized
plot_two_features <- function(df, feat1, feat2, doc_name='foo') {
	g <- ggplot(df, aes_string(x = feat1, y = feat2)) +
		geom_point(aes(color = contains_plag)) +
		ggtitle(doc_name)
	print(g)
}

# Reads data stored in <doc_path> and creates boxplots for each feature of the passages
# parsed out of <doc_path>
process_one_doc <- function(doc_path) {
	results <- read.csv(doc_path, head = T)
	results$contains_plag <- as.factor(results$contains_plag)
	
	
	if(length(levels(results$contains_plag)) > 1) {
		# Only plot those documents that contained plagiarism

		features <- c('average_word_length', 'stopword_percentage',
					  'punctuation_percentage', 'syntactic_complexity', 'avg_internal_word_freq_class',
					  'avg_external_word_freq_class')
					  
		for(feat in features) {
			out_file_name <- paste(strsplit(basename(doc_path), '.csv')[[1]], '_', feat, '.png', sep='')
			png(out_file_name)
			plot_one_metric(results, feat, doc_path)
			dev.off()
			print(paste('Wrote', out_file_name))
		}
	}
	
}

# Calls <process_one_doc> for all csv files found in <dir_name>
process_dir <- function(dir_name = '.') {
	all_files <- list.files(dir_name, pattern='*csv')
	for(f in all_files) {
		process_one_doc(f)
	}
}
