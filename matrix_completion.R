library(ggplot2)
library(reshape2)
library(cluster)
library(mclust)


set.seed(123)
n <- 100  
block_size <- n / 2
k <- 2

make_sbm <- function(N, clust, p, q) {
  A <- matrix(rbinom(N * N, 1, q), nrow = N, ncol = N)
  for (i in 1:length(clust)) {
    for (j in clust[[i]]) {
      for (k in clust[[i]]) {
        if (j != k) {
          A[j, k] <- rbinom(1, 1, p[i])
        }
      }
    }
  }
  diag(A) <- 0
  return(A)
}


clusters <- list(1:(n / 2), (n / 2 + 1):n) 
p <- c(0.3, 0.3) 
q <- 0.5    


A_full <- make_sbm(n, clusters, p, q)


A <- A_full
missing_indices <- sample(1:(n^2), size = floor(0.3 * n^2), replace = FALSE)
A[missing_indices] <- NA 


PMF <- function(A, k, lambda, learning_rate, num_epochs) {
  n <- nrow(A)
  m <- ncol(A)
  observed <- which(!is.na(A), arr.ind = TRUE)  
  
  U <- matrix(rnorm(n * k, sd = 0.1), n, k)
  V <- matrix(rnorm(m * k, sd = 0.1), m, k)
  
  #SGD 
  for (epoch in 1:num_epochs) {
    for (idx in 1:nrow(observed)) {
      i <- observed[idx, 1]
      j <- observed[idx, 2]
      error <- A[i, j] - sum(U[i, ] * V[j, ])
      
      #Update U and V using the gradient
      U[i, ] <- U[i, ] + learning_rate * (error * V[j, ] - lambda * U[i, ])
      V[j, ] <- V[j, ] + learning_rate * (error * U[i, ] - lambda * V[j, ])
    }
  }
  
  return(list(U = U, V = V))
}


lambda <- 0.1
learning_rate <- 0.01
num_epochs <- 1000
pmf_result <- PMF(A, k, lambda, learning_rate, num_epochs)

#matrix approximation
A_completed <- pmf_result$U %*% t(pmf_result$V)


rdpg_latent_positions <- function(A, d) {
  svd_result <- svd(A, nu = d, nv = d)
  latent_positions <- svd_result$u[, 1:d] %*% diag(sqrt(svd_result$d[1:d]))
  return(latent_positions)
}

latent_original <- rdpg_latent_positions(A_full, d = k)  # Original full matrix
latent_incomplete <- rdpg_latent_positions(replace(A, is.na(A), 0), d = k)  # Incomplete matrix
latent_completed <- rdpg_latent_positions(A_completed, d = k)  # Completed matrix

#kmeans
kmeans_original <- kmeans(latent_original, centers = 2)$cluster
kmeans_incomplete <- kmeans(latent_incomplete, centers = 2)$cluster
kmeans_completed <- kmeans(latent_completed, centers = 2)$cluster

plot_clusters <- function(latent_positions, clusters, title) {
  latent_df <- as.data.frame(latent_positions)
  latent_df$Cluster <- as.factor(clusters)
  ggplot(latent_df, aes(x = V1, y = V2, color = Cluster)) +
    geom_point(size = 3) +
    theme_minimal() +
    labs(title = title, x = "Latent Dimension 1", y = "Latent Dimension 2", color = "Cluster")
}

plot_clusters(latent_original, kmeans_original, "Clusters from Original A")
plot_clusters(latent_incomplete, kmeans_incomplete, "Clusters from Incomplete A")
plot_clusters(latent_completed, kmeans_completed, "Clusters from Completed A")


ground_truth <- c(rep(1, n / 2), rep(2, n / 2))

#GMM
gmm_original <- Mclust(latent_original, G = 2)  # G = 2 for 2 clusters
gmm_incomplete <- Mclust(latent_incomplete, G = 2)
gmm_completed <- Mclust(latent_completed, G = 2)


gmm_clusters_original <- gmm_original$classification
gmm_clusters_incomplete <- gmm_incomplete$classification
gmm_clusters_completed <- gmm_completed$classification

#ARI_GMM
ari_original_gmm <- adjustedRandIndex(ground_truth, gmm_clusters_original)
ari_incomplete_gmm <- adjustedRandIndex(ground_truth, gmm_clusters_incomplete)
ari_completed_gmm <- adjustedRandIndex(ground_truth, gmm_clusters_completed)


cat("ARI for Original A (GMM):", ari_original_gmm, "\n")
cat("ARI for Incomplete A (GMM):", ari_incomplete_gmm, "\n")
cat("ARI for Completed A (PMF, GMM):", ari_completed_gmm, "\n")

