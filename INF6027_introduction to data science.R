library(tidyverse)
library(dplyr)
library(nnet)
library(ggplot2)
install.packages("corrplot")
library(corrplot)
library(treemapify)
install.packages("caret")
library(caret)
install.packages("reshape2")
library(reshape2)

##PART 1 Pre Processing Data
#prepare dataset
spotify_data<-read.csv("D:/google download/dataset.csv")
head(spotify_data)

#Data Summary
#Test the distribution of popularity and delete parts of popularity data = 0
hist(spotify_data$popularity)
spotify_data<-filter(spotify_data,spotify_data$popularity!=0)
hist(spotify_data$popularity)


#select top 10 popular track genres
top_genres <- spotify_data %>%
  group_by(track_genre) %>%
  summarise(
    avg_popularity = mean(popularity, na.rm = TRUE),
    genre_count = n()
  ) %>%
  arrange(desc(avg_popularity)) %>%
  head(10) 
top_genre_names <- top_genres$track_genre
print(top_genre_names)

#recode track_genre to string type for further analyse
selected_spotify <- spotify_data %>%
  filter(track_genre %in% top_genre_names)
selected_spotify$genre <- as.factor(selected_spotify$track_genre)
str(selected_spotify)
levels(selected_spotify$genre)


ggplot(top_genres, aes(area = genre_count, fill = avg_popularity, label = track_genre)) +
  geom_treemap(linetype = "solid",colour = "white",size = 2) + 
  geom_treemap_text(fontface = "italic",colour = "white",place = "centre",size = 1 ,grow = TRUE) + 
  scale_fill_gradient(low = "lightblue", high = "darkblue") +  # 设置颜色渐变
  labs(title = "Tree Map of Top Genres",fill = "Average Popularity") +
  theme_minimal() 
  

#delete data except top 10 genres

topgenre_count<-selected_spotify %>%
  group_by(track_genre) %>%
  summarise(count=n())

ggplot(selected_spotify, aes(x = genre, fill = genre)) +
  geom_bar() +
  scale_fill_brewer(palette = "Set3") +  # 使用内置调色板 Set3
  labs(x = "Track Genre",y = "Count",title = "Top 10 Popular Genres Count")+
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, size = 3) +  # 添加计数值
  theme_minimal()

#change the length of song into minutes 
selected_spotify <- selected_spotify %>%
  mutate(duration_min = duration_ms / 60000)

#test multicollinearity by correlation matrix
feature_cols <- c("duration_min","danceability","energy","loudness","speechiness","acousticness","instrumentalness","liveness","valence")
cor_matrix <- cor(selected_spotify[feature_cols])
print(cor_matrix)


col <- colorRampPalette(c("lightblue", "white", "brown"))(200)
corrplot(cor_matrix, method = "circle", col = col, 
         addCoef.col = "black", number.cex = 0.7, 
         tl.cex = 1, tl.col = "black", 
         cl.lim = c(-1, 1), 
         tl.srt = 45)

#Use PCA to merge energy and valence
data_subset <- selected_spotify[, c("energy", "loudness")]
mood_scaled <- scale(data_subset)
pca_result <- prcomp(mood_scaled, center = TRUE, scale= TRUE)
summary(pca_result)
mood_pca <- pca_result$x[, 1:2]
selected_spotify<- cbind(selected_spotify, mood_pca)

#check multicollinearity in independent variables again
feature_cols2<- c("duration_min","PC1","danceability","speechiness","instrumentalness","liveness","valence")
cor_matrix2 <- cor(selected_spotify[feature_cols2])
print(cor_matrix2)
col <- colorRampPalette(c("lightblue", "white", "brown"))(200)
corrplot(cor_matrix2, method = "circle", col = col, 
         addCoef.col = "black", number.cex = 0.7, 
         tl.cex = 1, tl.col = "black", 
         cl.lim = c(-1, 1), 
         tl.srt = 45)

##PART2: build model and predict
#select reference category and build model
levels(selected_spotify$genre)
selected_spotify$genre <- relevel(selected_spotify$genre, ref = "chill")

spotify_multi_model <- multinom(genre ~  duration_min+ PC1 +danceability+ speechiness + instrumentalness + liveness +valence, 
                                data = selected_spotify)
summary(spotify_multi_model)
coefficients <- summary(spotify_multi_model)$coefficients
std_errors <- summary(spotify_multi_model)$standard.errors

#Use  Likelihood Ratio Test to measure the model fit
null_model <- multinom(genre ~ 1, data = selected_spotify)
anova(null_model, spotify_multi_model, test = "Chisq")

#use visualization way to show coefficient
coef_df <- as.data.frame(t(coefficients))
coef_df$Feature <- colnames(coefficients)
coef_df <- coef_df %>% 
  pivot_longer(-Feature, names_to = "Genre", values_to = "Coefficient")
ggplot(coef_df, aes(x = Genre, y = Coefficient, fill = Feature)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Feature Effects on Music Genres",
       x = "Genres",
       y = "Coefficient") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

##Build prediction model
set.seed(123)
trainIndex <- createDataPartition(selected_spotify$genre, p = 0.8, list = FALSE)
train_data <- selected_spotify[trainIndex, ]
test_data <- selected_spotify[-trainIndex, ]
spotify_predict_model <- multinom(genre ~ duration_min+PC1+danceability+ speechiness + instrumentalness + liveness +valence, data = train_data)
genre_probability<-predict(spotify_predict_model,newdata=test_data,type='probs')
head(genre_probability)

#use heatmap to show probability result
genre_prob_df <- as.data.frame(genre_probability)
colnames(genre_prob_df) <- colnames(genre_probability)
genre_prob_long <- melt(genre_prob_df)
genre_prob_long$SampleID <- rep(1:nrow(genre_prob_df), times = ncol(genre_prob_df))
ggplot(genre_prob_long, aes(x = SampleID, y = variable, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "darkblue") +
  labs(x = 'Sample ID', y = 'Genre', fill = 'Probability') +
  theme_minimal()

#calculate the average probability of ten genres and rank them 
avg_prob <- colMeans(genre_probability)
avg_prob_sorted <- sort(avg_prob, decreasing = TRUE)
avg_prob_sorted

#Build confusion matrix to evaluate the whole model
genre_predicted <- colnames(genre_probability)[max.col(genre_probability, ties.method = "first")]
is.factor(genre_predicted) 
is.factor(test_data$genre)  
genre_predicted <- factor(genre_predicted)
conf_matrix <- confusionMatrix(genre_predicted , test_data$genre)
print(conf_matrix)
accuracy <- conf_matrix$overall['Accuracy']  
print(accuracy)

#create confusion matrix heat map
conf_matrix_df <- as.data.frame(as.table(conf_matrix$table))
ggplot(conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix Heatmap", x = "Predicted", y = "Actual", fill = "Frequency") +
  theme_minimal()
