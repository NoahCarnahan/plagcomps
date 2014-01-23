library(ggplot2)
library(Hmisc)


preprocess_frame <- function(df, conf_cutoff = .35) {
  df$contains_plag <- as.factor(df$contains_plag)
  df$passage_length <- df$end_index - df$start_index
  df$plag_predicted <- ifelse(df$plag_confidence > conf_cutoff, 1, 0)
  df$true_pos <- ifelse(df$plag_predicted == 1 & df$contains_plag == 1, 1, 0)
  df$true_neg <- ifelse(df$plag_predicted == 0 & df$contains_plag == 0, 1, 0)
  
  df$false_pos <- ifelse(df$plag_predicted == 1 & df$contains_plag == 0, 1, 0)
  df$false_neg <- ifelse(df$plag_predicted == 0 & df$contains_plag == 1, 1, 0)
  
  df$result_of_class <- as.factor(ifelse(df$true_pos == 1, 'true_pos', 
                               ifelse(df$true_neg == 1, 'true_neg',
                                      ifelse(df$false_pos == 1, 'false_pos',
                                             ifelse(df$false_neg == 1, 'false_neg', 'NONE')))))
  
  df$how_wrong <- ifelse(df$contains_plag == 1, 1 - df$plag_confidence, df$plag_confidence)
  
  return(df)
  
}

plot_which_wrong <- function(df, doc_name='foo') {
  g <- ggplot(data = df, aes(x = how_wrong, y = passage_length)) +
    geom_point() +
    ggtitle(doc_name)
  
  print(g)
}


# Boxplot of <metric> broken down by whether the passage was plagiarized
plot_one_metric <- function(df, metric, doc_name='foo') {
  # t-test: null hypothesis that plag./non plag. passages have same dist of <metric>
  # reject with small p-value suggests that <metric> is different between plag./non plag. passages
  formula <- paste(metric, '~', 'contains_plag')
  ttest <- t.test(as.formula(formula), data = df)
  pval <- ttest$p.value
  
  g <- ggplot(df) +
    geom_boxplot(aes_string(x = 'contains_plag', y = metric)) +
    scale_x_discrete('Contains Plag?', labels = c('No', 'Yes')) +
    ggtitle(paste(doc_name, '(With pval =', pval, ')'))
  print(g)  
}

# Scatterplot of feat1 on x-axis, feat2 on y-axis and points 
# colored by whether or not the passage was plagiarized
plot_two_features <- function(df, feat1, feat2, doc_name='foo') {
	g <- ggplot(df, aes_string(x = feat1, y = feat2)) +
		geom_point(aes(color = contains_plag, size = plag_confidence)) +
		ggtitle(doc_name)
	print(g)
}

# Reads data stored in <doc_path> and creates boxplots for each feature of the passages
# parsed out of <doc_path>
boxplot_one_doc <- function(doc_path) {
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

# Reads data stored in <doc_path> and creates boxplots for each feature of the passages
# parsed out of <doc_path>
scatter_one_doc <- function(doc_path, xvar = 'syntactic_complexity', yvar = 'avg_external_word_freq_class') {
  results <- read.csv(doc_path, head = T)
  results$contains_plag <- as.factor(results$contains_plag)
  
  
  if(length(levels(results$contains_plag)) > 1) {
    # Only plot those documents that contained plagiarism
    
    out_file_name <- paste(strsplit(basename(doc_path), '.csv')[[1]], '_', yvar, '_', xvar, '.png', sep='')
    print(out_file_name)
    png(out_file_name)
    plot_two_features(results, xvar, yvar, doc_name=doc_path)
    dev.off()
    print(paste('Wrote', out_file_name))
    
  }
  
}

how_wrong_one_doc <- function(doc_path) {
  results <- read.csv(doc_path, head = T)
  processed <- preprocess_frame(results)
  out_file_name <- paste(strsplit(basename(doc_path), '.csv')[[1]], '_how_wrong', '.pdf', sep='')
  print(out_file_name)
  
  pdf(out_file_name)
  plot_which_wrong(processed, doc_path)
  dev.off()
}

# Calls <boxplot_one_doc> for all csv files found in <dir_name>
process_dir <- function(dir_name = '.', plot_type = 'scatter') {
	all_files <- list.files(dir_name, pattern='*csv')
  
  
	 for(f in all_files) {
	   if(plot_type == 'boxplot') {
       boxplot_one_doc(f)
      } else if(plot_type == 'scatter') {
        scatter_one_doc(f)
      } else if(plot_type == 'how_wrong') {
        how_wrong_one_doc(f)
      }
  }
}
