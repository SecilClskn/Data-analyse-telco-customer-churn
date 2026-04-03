# =============================================================================
# ANALYSE COMPLETE - TELCO CUSTOMER CHURN  (VERSION AMÉLIORÉE)
# Corrections : suppression NA, data leakage TotalCharges, clustering enrichi,
# LR baseline, upsampling, recall, test de Schoenfeld, langage causal
# =============================================================================

# ---- 0. INSTALLER ET CHARGER LES PACKAGES ----------------------------------

packages_necessaires <- c(
  "tidyverse", "ggplot2", "corrplot", "caret", "e1071",
  "randomForest", "nnet", "survival", "survminer", "cluster",
  "factoextra", "gridExtra", "scales", "RColorBrewer",
  "pROC", "reshape2", "kernlab"
)

options(repos = c(CRAN = "https://cloud.r-project.org"))

for (p in packages_necessaires) {
  if (!require(p, character.only = TRUE, quietly = TRUE)) {
    install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}

cat("Tous les packages sont chargés !\n")

if (!dir.exists("graphiques")) dir.create("graphiques")

# Fonction utilitaire pour sauvegarder les graphiques
sauvegarder_graph <- function(nom_fichier, largeur = 10, hauteur = 7) {
  ggsave(paste0("graphiques/", nom_fichier, ".png"),
         width = largeur, height = hauteur, dpi = 150, bg = "white")
  cat("  -> Graphique sauvegardé :", nom_fichier, "\n")
}


# =============================================================================
# PARTIE 1 : CHARGEMENT ET NETTOYAGE
# =============================================================================
cat("\n========== PARTIE 1 : CHARGEMENT ET NETTOYAGE ==========\n")

donnees <- read.csv(
  "WA_Fn-UseC_-Telco-Customer-Churn.csv",
  stringsAsFactors = FALSE,
  na.strings = c("", " ", "NA")
)

# Convertir TotalCharges en numérique (il y a des espaces -> NA)
donnees$TotalCharges <- as.numeric(donnees$TotalCharges)

cat("Valeurs manquantes :\n")
print(colSums(is.na(donnees))[colSums(is.na(donnees)) > 0])

# ---- 1.1 Suppression des 11 lignes NA --------------------------------------
# On SUPPRIME les 11 lignes (toutes avec tenure=0, soit 0.16% du dataset).
# L'imputation TotalCharges = tenure × MonthlyCharges serait une approximation
# discutable (ignore remises, frais d'installation, etc.).
# 11 lignes sur 7043 = négligeable, la suppression est plus propre.

n_avant <- nrow(donnees)
donnees  <- donnees[!is.na(donnees$TotalCharges), ]
cat(sprintf("Lignes supprimées (TotalCharges NA) : %d (%.2f%%)\n",
            n_avant - nrow(donnees),
            (n_avant - nrow(donnees)) / n_avant * 100))

# ---- 1.2 Conversions de types ----------------------------------------------
donnees$Churn_bin <- ifelse(donnees$Churn == "Yes", 1, 0)
donnees$Churn     <- as.factor(donnees$Churn)

# SeniorCitizen : 0/1 -> facteur lisible
donnees$SeniorCitizen <- factor(donnees$SeniorCitizen,
                                 levels = c(0, 1), labels = c("No", "Yes"))

colonnes_texte <- c(
  "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
  "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
  "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
  "PaperlessBilling", "PaymentMethod"
)
donnees[colonnes_texte] <- lapply(donnees[colonnes_texte], as.factor)

# Supprimer l'ID client
donnees_clean <- donnees[, !names(donnees) %in% "customerID"]

cat("Dataset final :", nrow(donnees_clean), "x", ncol(donnees_clean), "\n")


# =============================================================================
# PARTIE 2 : STATISTIQUES DESCRIPTIVES
# =============================================================================
cat("\n========== PARTIE 2 : STATISTIQUES DESCRIPTIVES ==========\n")

cat("\nRésumé numérique :\n")
print(summary(donnees_clean[, c("tenure", "MonthlyCharges", "TotalCharges")]))

taux_churn <- mean(donnees_clean$Churn_bin) * 100
cat(sprintf("Taux de churn : %.1f%%\n", taux_churn))

# ---- GRAPHIQUE 01 : Distribution du Churn ----------------------------------
df_churn <- data.frame(
  Churn = c("Not Churned", "Churned"),
  Count = c(sum(donnees_clean$Churn == "No"), sum(donnees_clean$Churn == "Yes"))
)
df_churn$Pct   <- round(df_churn$Count / sum(df_churn$Count) * 100, 1)
df_churn$Label <- paste0(df_churn$Pct, "%\n(n=", df_churn$Count, ")")

ggplot(df_churn, aes(x = Churn, y = Count, fill = Churn)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_text(aes(label = Label), vjust = -0.3, size = 5) +
  scale_fill_manual(values = c("#2196F3", "#F44336")) +
  labs(title = "Customer Churn Distribution",
       subtitle = paste0("Overall churn rate: ", round(taux_churn, 1), "%"),
       x = "Churn Status", y = "Number of Customers") +
  theme_minimal(base_size = 13) + theme(legend.position = "none") +
  ylim(0, max(df_churn$Count) * 1.15)
sauvegarder_graph("01_distribution_churn")

# ---- GRAPHIQUE 02 : Distribution tenure ------------------------------------
ggplot(donnees_clean, aes(x = tenure, fill = Churn)) +
  geom_histogram(binwidth = 3, color = "white", alpha = 0.8) +
  scale_fill_manual(values = c("#2196F3", "#F44336"),
                    labels = c("Not Churned", "Churned")) +
  labs(title = "Customer Tenure Distribution",
       x = "Tenure (months)", y = "Number of Customers", fill = "Status") +
  theme_minimal(base_size = 13) +
  facet_wrap(~Churn, labeller = labeller(Churn = c(No = "Not Churned", Yes = "Churned")))
sauvegarder_graph("02_distribution_tenure")

# ---- GRAPHIQUE 03 : Boxplots variables numériques --------------------------
p1 <- ggplot(donnees_clean, aes(x = Churn, y = tenure, fill = Churn)) +
  geom_boxplot(alpha = 0.7) + scale_fill_manual(values = c("#2196F3","#F44336")) +
  labs(title = "Tenure", x = "Churn", y = "Months") +
  theme_minimal(base_size = 12) + theme(legend.position = "none")

p2 <- ggplot(donnees_clean, aes(x = Churn, y = MonthlyCharges, fill = Churn)) +
  geom_boxplot(alpha = 0.7) + scale_fill_manual(values = c("#2196F3","#F44336")) +
  labs(title = "Monthly Charges", x = "Churn", y = "Dollars") +
  theme_minimal(base_size = 12) + theme(legend.position = "none")

p3 <- ggplot(donnees_clean, aes(x = Churn, y = TotalCharges, fill = Churn)) +
  geom_boxplot(alpha = 0.7) + scale_fill_manual(values = c("#2196F3","#F44336")) +
  labs(title = "Total Charges", x = "Churn", y = "Dollars") +
  theme_minimal(base_size = 12) + theme(legend.position = "none")

grid.arrange(p1, p2, p3, ncol = 3,
             top = "Numerical Variables by Churn Status")
sauvegarder_graph("03_boxplots_numeriques")

# ---- GRAPHIQUE 04 : Churn par contrat --------------------------------------
df_contrat <- donnees_clean %>%
  group_by(Contract, Churn) %>% summarise(n = n(), .groups = "drop") %>%
  group_by(Contract) %>% mutate(pct = n / sum(n) * 100)

ggplot(df_contrat, aes(x = Contract, y = pct, fill = Churn)) +
  geom_bar(stat = "identity", position = "stack") +
  geom_text(aes(label = paste0(round(pct,1),"%")),
            position = position_stack(vjust = 0.5), color = "white", size = 4.5) +
  scale_fill_manual(values = c("#2196F3","#F44336"),
                    labels = c("Not Churned","Churned")) +
  labs(title = "Churn Rate by Contract Type",
       x = "Contract Type", y = "Proportion (%)", fill = "Status") +
  theme_minimal(base_size = 13)
sauvegarder_graph("04_churn_par_contrat")

# ---- GRAPHIQUE 05 : Churn par Internet -------------------------------------
df_internet <- donnees_clean %>%
  group_by(InternetService, Churn) %>% summarise(n = n(), .groups = "drop") %>%
  group_by(InternetService) %>% mutate(pct = n / sum(n) * 100)

ggplot(df_internet, aes(x = InternetService, y = pct, fill = Churn)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = paste0(round(pct,1),"%")),
            position = position_dodge(width = 0.9), vjust = -0.3, size = 4) +
  scale_fill_manual(values = c("#2196F3","#F44336"),
                    labels = c("Not Churned","Churned")) +
  labs(title = "Churn by Internet Service Type",
       x = "Internet Service", y = "Proportion (%)", fill = "Status") +
  theme_minimal(base_size = 13) + ylim(0, 100)
sauvegarder_graph("05_churn_internet")

# ---- GRAPHIQUE 06 : Churn par méthode de paiement --------------------------
df_payment <- donnees_clean %>%
  group_by(PaymentMethod, Churn) %>% summarise(n = n(), .groups = "drop") %>%
  group_by(PaymentMethod) %>% mutate(pct = n / sum(n) * 100) %>%
  filter(Churn == "Yes")

ggplot(df_payment, aes(x = reorder(PaymentMethod, pct), y = pct, fill = pct)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(pct,1),"%")), hjust = -0.1, size = 4.5) +
  scale_fill_gradient(low = "#90CAF9", high = "#D32F2F") +
  coord_flip() +
  labs(title = "Churn Rate by Payment Method", x = "", y = "Churn Rate (%)") +
  theme_minimal(base_size = 13) + theme(legend.position = "none") + ylim(0, 60)
sauvegarder_graph("06_churn_paiement")

# ---- GRAPHIQUE 07 : Variables démographiques --------------------------------
vars_demo   <- c("gender", "SeniorCitizen", "Partner", "Dependents")
labels_demo <- c("Gender", "Senior Citizen", "Partner", "Dependents")

plots_demo <- lapply(seq_along(vars_demo), function(i) {
  df_temp <- donnees_clean %>%
    group_by(.data[[vars_demo[i]]], Churn) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(.data[[vars_demo[i]]]) %>%
    mutate(pct = n / sum(n) * 100)
  ggplot(df_temp, aes(x = .data[[vars_demo[i]]], y = pct, fill = Churn)) +
    geom_bar(stat = "identity", position = "fill") +
    scale_fill_manual(values = c("#2196F3","#F44336"),
                      labels = c("Not Churned","Churned")) +
    scale_y_continuous(labels = scales::percent) +
    labs(title = labels_demo[i], x = "", y = "") +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom", legend.title = element_blank())
})
grid.arrange(grobs = plots_demo, ncol = 2,
             top = "Churn by Demographic Variables")
sauvegarder_graph("07_churn_demographie")

# ---- GRAPHIQUE 08 : Heatmap services ----------------------------------------
services <- c("OnlineSecurity","OnlineBackup","DeviceProtection",
              "TechSupport","StreamingTV","StreamingMovies")

df_svc <- lapply(services, function(s) {
  donnees_clean %>%
    group_by(Service = .data[[s]], Churn) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(Service) %>%
    mutate(pct = n / sum(n) * 100, Variable = s) %>%
    filter(Churn == "Yes")
}) %>% bind_rows()

ggplot(df_svc, aes(x = Variable, y = Service, fill = pct)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = paste0(round(pct,1),"%")), color = "white", size = 4) +
  scale_fill_gradient2(low = "#1565C0", mid = "#FFC107", high = "#B71C1C",
                       midpoint = 30, name = "Churn\nRate (%)") +
  labs(title = "Churn Rate by Subscribed Services",
       x = "Service", y = "Service Level") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
sauvegarder_graph("08_heatmap_services")

# ---- GRAPHIQUE 09 : Corrélation --------------------------------------------
# Note : TotalCharges est inclus ici pour l'EDA (r=0.83 avec tenure visible)
# mais sera EXCLU du pipeline ML (data leakage / redondance)
mat_cor <- cor(donnees_clean[, c("tenure","MonthlyCharges","TotalCharges","Churn_bin")])
colnames(mat_cor) <- rownames(mat_cor) <- c("Tenure","Monthly\nCharges",
                                             "Total\nCharges","Churn")
png("graphiques/09_correlation.png", width = 800, height = 700, res = 120)
corrplot(mat_cor, method = "color", type = "upper", addCoef.col = "black",
         tl.col = "black", tl.srt = 30,
         col = colorRampPalette(c("#1565C0","white","#B71C1C"))(200),
         title = "Correlation Matrix - Numerical Variables\n(TotalCharges excluded from ML)",
         mar = c(0, 0, 3, 0))
dev.off()
cat("  -> Graphique sauvegardé : 09_correlation\n")

# ---- GRAPHIQUE 10 : Scatter Tenure vs Monthly Charges ----------------------
ggplot(donnees_clean, aes(x = tenure, y = MonthlyCharges, color = Churn)) +
  geom_point(alpha = 0.4, size = 1.5) +
  geom_smooth(method = "loess", se = TRUE) +
  scale_color_manual(values = c("#2196F3","#F44336"),
                     labels = c("Not Churned","Churned")) +
  labs(title = "Monthly Charges vs Tenure by Churn Status",
       x = "Tenure (months)", y = "Monthly Charges ($)", color = "Status") +
  theme_minimal(base_size = 13)
sauvegarder_graph("10_scatter_tenure_charges")


# =============================================================================
# PARTIE 3 : PRÉPARATION DES FEATURES POUR LE ML
# =============================================================================
cat("\n========== PARTIE 3 : PRÉPARATION DES FEATURES ==========\n")

# Encodage dummy des variables catégorielles
donnees_ml <- donnees_clean %>%
  select(-Churn_bin) %>%
  mutate(across(where(is.factor), as.character))

mat_dummy <- model.matrix(~ . - Churn - 1, data = donnees_ml)
mat_dummy <- as.data.frame(mat_dummy)

# Ajouter la cible
mat_dummy$Churn <- donnees_clean$Churn

# ---- Suppression de TotalCharges du pipeline ML ----------------------------
# Raison : TotalCharges ≈ tenure × MonthlyCharges (r=0.83 avec tenure).
# Garder les 3 variables crée une redondance qui biaise l'importance des
# variables et peut poser un problème de data leakage (TotalCharges encode
# partiellement la durée d'abonnement, i.e. la variable cible survie).
# On conserve tenure + MonthlyCharges comme représentations indépendantes.
if ("TotalCharges" %in% names(mat_dummy)) {
  mat_dummy <- mat_dummy %>% select(-TotalCharges)
  cat("TotalCharges exclu du pipeline ML (data leakage / colinéarité).\n")
}

# Normalisation des variables numériques restantes
cols_num <- c("tenure", "MonthlyCharges")
for (col in cols_num) {
  mat_dummy[[col]] <- scale(mat_dummy[[col]])[, 1]
}

# Nettoyage des noms de colonnes
names(mat_dummy) <- make.names(names(mat_dummy))
cat("Dimensions de la matrice ML :", nrow(mat_dummy), "x", ncol(mat_dummy), "\n")

# Division 70/30 stratifiée
set.seed(42)
index_train <- createDataPartition(mat_dummy$Churn, p = 0.70, list = FALSE)
train_data  <- mat_dummy[index_train, ]
test_data   <- mat_dummy[-index_train, ]
cat("Train:", nrow(train_data), "| Test:", nrow(test_data), "\n")


# =============================================================================
# PARTIE 4 : CLUSTERING (K-Means enrichi sur toutes les features)
# =============================================================================
cat("\n========== PARTIE 4 : CLUSTERING K-MEANS ==========\n")

# Utiliser toutes les features (sans Churn) pour un clustering plus riche.
# Les variables sont déjà centrées/réduites -> pas de re-normalisation.
donnees_cluster <- mat_dummy %>% select(-Churn)

# ---- GRAPHIQUE 11 : Variance expliquée par PCA (scree plot) ----------------
# On fait une PCA pour comprendre la structure des données avant le clustering
pca_res   <- prcomp(donnees_cluster, center = FALSE, scale. = FALSE)
var_exp   <- pca_res$sdev^2 / sum(pca_res$sdev^2) * 100
var_cum   <- cumsum(var_exp)
df_scree  <- data.frame(PC = 1:min(20, length(var_exp)),
                         Var = var_exp[1:min(20, length(var_exp))],
                         Cum = var_cum[1:min(20, length(var_exp))])

ggplot(df_scree, aes(x = PC)) +
  geom_bar(aes(y = Var), stat = "identity", fill = "#1565C0", alpha = 0.7) +
  geom_line(aes(y = Cum), color = "#F44336", linewidth = 1.2) +
  geom_point(aes(y = Cum), color = "#F44336", size = 3) +
  geom_hline(yintercept = 70, linetype = "dashed", color = "grey50") +
  annotate("text", x = 15, y = 72, label = "70% threshold", color = "grey40", size = 3.5) +
  scale_y_continuous(name = "Variance Explained (%)",
                     sec.axis = sec_axis(~., name = "Cumulative Variance (%)")) +
  labs(title = "PCA Scree Plot - Feature Structure",
       subtitle = "Bars: individual variance | Line: cumulative variance",
       x = "Principal Component") +
  theme_minimal(base_size = 12)
sauvegarder_graph("11_pca_scree")

# ---- GRAPHIQUE 12 : Méthode du coude ----------------------------------------
set.seed(42)
inertie <- sapply(1:8, function(k) {
  kmeans(donnees_cluster, centers = k, nstart = 25, iter.max = 100)$tot.withinss
})

df_coude <- data.frame(k = 1:8, inertie = inertie)

ggplot(df_coude, aes(x = k, y = inertie)) +
  geom_line(color = "#1565C0", linewidth = 1.2) +
  geom_point(color = "#F44336", size = 4) +
  geom_vline(xintercept = 3, linetype = "dashed", color = "grey50") +
  annotate("text", x = 3.3, y = max(inertie) * 0.9,
           label = "k=3 (chosen)", color = "grey40", size = 4) +
  labs(title = "Elbow Method - Optimal Number of Clusters",
       subtitle = "K-means on all encoded features (after removing TotalCharges)",
       x = "Number of Clusters (k)", y = "Within-Cluster Inertia") +
  theme_minimal(base_size = 13)
sauvegarder_graph("12_methode_coude")

# K-Means k=3 sur toutes les features
set.seed(42)
km_model <- kmeans(donnees_cluster, centers = 3, nstart = 50, iter.max = 200)
donnees_clean$Cluster <- as.factor(km_model$cluster)

cat("\nTaille des clusters :", table(donnees_clean$Cluster), "\n")

# ---- GRAPHIQUE 13 : Visualisation PCA des clusters --------------------------
fviz_cluster(km_model, data = donnees_cluster,
             palette = c("#F44336","#2196F3","#4CAF50"),
             geom = "point", ellipse.type = "norm",
             ggtheme = theme_minimal(base_size = 13)) +
  labs(title = "Customer Clusters (PCA Projection)",
       subtitle = "K-means on all features (30 encoded variables)")
sauvegarder_graph("13_clusters_pca")

# ---- GRAPHIQUE 14 : Profil des clusters -------------------------------------
df_profil <- donnees_clean %>%
  group_by(Cluster) %>%
  summarise(
    Avg_Tenure  = mean(tenure),
    Avg_Charges = mean(MonthlyCharges),
    Churn_Rate  = mean(Churn_bin) * 100,
    .groups = "drop"
  )

df_profil_long <- melt(as.data.frame(df_profil), id.vars = "Cluster")

ggplot(df_profil_long, aes(x = Cluster, y = value, fill = Cluster)) +
  geom_bar(stat = "identity") +
  facet_wrap(~variable, scales = "free_y",
             labeller = labeller(variable = c(
               Avg_Tenure  = "Avg. Tenure (months)",
               Avg_Charges = "Avg. Monthly Charges ($)",
               Churn_Rate  = "Churn Rate (%)"
             ))) +
  scale_fill_manual(values = c("#F44336","#2196F3","#4CAF50")) +
  labs(title = "Profile of 3 Customer Clusters", x = "Cluster", y = "Value") +
  theme_minimal(base_size = 12) + theme(legend.position = "none")
sauvegarder_graph("14_profil_clusters")

# ---- GRAPHIQUE 15 : Churn par cluster et contrat ----------------------------
ggplot(donnees_clean, aes(x = Cluster, fill = Churn)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("#2196F3","#F44336"),
                    labels = c("Not Churned","Churned")) +
  facet_wrap(~Contract) +
  labs(title = "Churn Rate by Cluster and Contract Type",
       x = "Cluster", y = "Proportion", fill = "Status") +
  theme_minimal(base_size = 12)
sauvegarder_graph("15_churn_cluster_contrat")


# =============================================================================
# PARTIE 5 : MODÈLES DE CLASSIFICATION
# =============================================================================

# Paramètres de validation croisée avec upsampling
# sampling = "up" : sur-échantillonne la classe minoritaire (churners, 26.5%)
# dans chaque fold d'entraînement pour améliorer le recall.
# L'évaluation reste sur le test set original (non rééchantillonné).
ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling        = "up"   # correction du déséquilibre de classes
)

# -----------------------------------------------------------------
# 5.1 : RÉGRESSION LOGISTIQUE (baseline)
# -----------------------------------------------------------------
cat("\n========== MODÈLE 0 : LOGISTIC REGRESSION (BASELINE) ==========\n")
set.seed(42)

lr_model <- train(
  Churn ~ .,
  data      = train_data,
  method    = "glm",
  family    = "binomial",
  trControl = ctrl,
  metric    = "ROC"
)

lr_pred      <- predict(lr_model, test_data)
lr_pred_prob <- predict(lr_model, test_data, type = "prob")[, "Yes"]
cm_lr <- confusionMatrix(lr_pred, test_data$Churn, positive = "Yes")
cat("Accuracy LR:", round(cm_lr$overall["Accuracy"] * 100, 2), "%\n")
cat("Recall  LR:", round(cm_lr$byClass["Recall"] * 100, 2), "%\n")

# ---- GRAPHIQUE 16 : Importance des variables - Logistic Regression ---------
imp_lr <- varImp(lr_model)$importance
imp_lr$Variable <- rownames(imp_lr)

top20_lr <- imp_lr %>% arrange(desc(Overall)) %>% head(20)

ggplot(top20_lr, aes(x = reorder(Variable, Overall), y = Overall, fill = Overall)) +
  geom_bar(stat = "identity") + coord_flip() +
  scale_fill_gradient(low = "#A5D6A7", high = "#1B5E20") +
  labs(title = "Top 20 Variables - Logistic Regression (Baseline)",
       x = "", y = "Importance (|z-score|)") +
  theme_minimal(base_size = 11) + theme(legend.position = "none")
sauvegarder_graph("16_importance_logistic")

# -----------------------------------------------------------------
# 5.2 : RANDOM FOREST
# -----------------------------------------------------------------
cat("\n========== MODÈLE 1 : RANDOM FOREST ==========\n")
set.seed(42)

rf_model <- train(
  Churn ~ .,
  data      = train_data,
  method    = "rf",
  trControl = ctrl,
  metric    = "ROC",
  tuneGrid  = expand.grid(mtry = c(5, 10, 15)),
  ntree     = 200
)

rf_pred      <- predict(rf_model, test_data)
rf_pred_prob <- predict(rf_model, test_data, type = "prob")[, "Yes"]
cm_rf <- confusionMatrix(rf_pred, test_data$Churn, positive = "Yes")
cat("Accuracy RF:", round(cm_rf$overall["Accuracy"] * 100, 2), "%\n")
cat("Recall  RF:", round(cm_rf$byClass["Recall"] * 100, 2), "%\n")

# ---- GRAPHIQUE 17 : Importance des variables - Random Forest ---------------
imp_rf <- varImp(rf_model)$importance
imp_rf$Variable <- rownames(imp_rf)

top20_rf <- imp_rf %>% arrange(desc(Overall)) %>% head(20)

ggplot(top20_rf, aes(x = reorder(Variable, Overall), y = Overall, fill = Overall)) +
  geom_bar(stat = "identity") + coord_flip() +
  scale_fill_gradient(low = "#90CAF9", high = "#0D47A1") +
  labs(title = "Top 20 Important Variables - Random Forest",
       x = "", y = "Importance (Mean Decrease Gini)") +
  theme_minimal(base_size = 11) + theme(legend.position = "none")
sauvegarder_graph("17_importance_random_forest")

# -----------------------------------------------------------------
# 5.3 : SVM
# -----------------------------------------------------------------
cat("\n========== MODÈLE 2 : SVM ==========\n")
set.seed(42)

svm_model <- train(
  Churn ~ .,
  data       = train_data,
  method     = "svmRadial",
  trControl  = ctrl,
  metric     = "ROC",
  tuneGrid   = expand.grid(C = c(0.1, 1, 10), sigma = c(0.01, 0.05)),
  preProcess = c("center", "scale")
)

svm_pred      <- predict(svm_model, test_data)
svm_pred_prob <- predict(svm_model, test_data, type = "prob")[, "Yes"]
cm_svm <- confusionMatrix(svm_pred, test_data$Churn, positive = "Yes")
cat("Accuracy SVM:", round(cm_svm$overall["Accuracy"] * 100, 2), "%\n")
cat("Recall  SVM:", round(cm_svm$byClass["Recall"] * 100, 2), "%\n")

# ---- GRAPHIQUE 18 : Hyperparamètres SVM ------------------------------------
ggplot(svm_model$results, aes(x = factor(C), y = ROC,
       color = factor(sigma), group = factor(sigma))) +
  geom_line(linewidth = 1.2) + geom_point(size = 3) +
  scale_color_brewer(palette = "Set1", name = "Sigma") +
  labs(title = "SVM Optimization - ROC by Hyperparameters",
       x = "Cost C", y = "AUC-ROC") +
  theme_minimal(base_size = 13)
sauvegarder_graph("18_svm_hyperparametres")

# -----------------------------------------------------------------
# 5.4 : RÉSEAU DE NEURONES
# -----------------------------------------------------------------
cat("\n========== MODÈLE 3 : NEURAL NETWORK ==========\n")
set.seed(42)

nn_model <- train(
  Churn ~ .,
  data       = train_data,
  method     = "nnet",
  trControl  = ctrl,
  metric     = "ROC",
  tuneGrid   = expand.grid(size = c(5, 10, 15), decay = c(0.001, 0.01, 0.1)),
  maxit      = 300,
  trace      = FALSE,
  preProcess = c("center", "scale")
)

nn_pred      <- predict(nn_model, test_data)
nn_pred_prob <- predict(nn_model, test_data, type = "prob")[, "Yes"]
cm_nn <- confusionMatrix(nn_pred, test_data$Churn, positive = "Yes")
cat("Accuracy NN:", round(cm_nn$overall["Accuracy"] * 100, 2), "%\n")
cat("Recall  NN:", round(cm_nn$byClass["Recall"] * 100, 2), "%\n")

# ---- GRAPHIQUE 19 : Hyperparamètres Neural Network -------------------------
ggplot(nn_model$results, aes(x = factor(size), y = ROC,
       color = factor(decay), group = factor(decay))) +
  geom_line(linewidth = 1.2) + geom_point(size = 3) +
  scale_color_brewer(palette = "Dark2", name = "Decay") +
  labs(title = "Neural Network Optimization - ROC",
       x = "Number of Hidden Neurons", y = "AUC-ROC") +
  theme_minimal(base_size = 13)
sauvegarder_graph("19_nn_hyperparametres")


# =============================================================================
# PARTIE 6 : COMPARAISON DES 4 MODÈLES
# =============================================================================
cat("\n========== PARTIE 6 : COMPARAISON DES MODÈLES ==========\n")

# Calcul des courbes ROC
roc_lr  <- roc(as.numeric(test_data$Churn == "Yes"), lr_pred_prob,  quiet = TRUE)
roc_rf  <- roc(as.numeric(test_data$Churn == "Yes"), rf_pred_prob,  quiet = TRUE)
roc_svm <- roc(as.numeric(test_data$Churn == "Yes"), svm_pred_prob, quiet = TRUE)
roc_nn  <- roc(as.numeric(test_data$Churn == "Yes"), nn_pred_prob,  quiet = TRUE)

auc_lr  <- round(auc(roc_lr),  4)
auc_rf  <- round(auc(roc_rf),  4)
auc_svm <- round(auc(roc_svm), 4)
auc_nn  <- round(auc(roc_nn),  4)

cat(sprintf("AUC Logistic Regression : %.4f\n", auc_lr))
cat(sprintf("AUC Random Forest       : %.4f\n", auc_rf))
cat(sprintf("AUC SVM                 : %.4f\n", auc_svm))
cat(sprintf("AUC Neural Network      : %.4f\n", auc_nn))

# ---- GRAPHIQUE 20 : Courbes ROC (4 modèles) --------------------------------
make_roc_df <- function(roc_obj, label) {
  data.frame(FPR = 1 - roc_obj$specificities,
             TPR = roc_obj$sensitivities,
             Model = label)
}
df_roc_all <- rbind(
  make_roc_df(roc_lr,  paste0("Logistic Reg. (AUC=", auc_lr,  ")")),
  make_roc_df(roc_rf,  paste0("Random Forest (AUC=", auc_rf,  ")")),
  make_roc_df(roc_svm, paste0("SVM (AUC=",           auc_svm, ")")),
  make_roc_df(roc_nn,  paste0("Neural Network (AUC=",auc_nn,  ")"))
)

ggplot(df_roc_all, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c("#9C27B0","#4CAF50","#F44336","#2196F3")) +
  labs(title = "ROC Curves - Comparison of 4 Models",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       color = "Model") +
  theme_minimal(base_size = 13) + theme(legend.position = "bottom")
sauvegarder_graph("20_courbes_roc", largeur = 10, hauteur = 7)

# ---- GRAPHIQUE 21 : Comparaison métriques (4 modèles) ----------------------
# Note : le recall (~50-60%) est intentionnellement bas sans SMOTE/class weights plus agressifs.
# Dans un contexte métier, on préférerait maximiser le recall (détecter max de churners)
# au détriment de la précision. L'upsampling déjà appliqué améliore le recall vs baseline.
metriques <- data.frame(
  Model     = c("Logistic Reg.", "Random Forest", "SVM", "Neural Network"),
  Accuracy  = c(cm_lr$overall["Accuracy"], cm_rf$overall["Accuracy"],
                cm_svm$overall["Accuracy"], cm_nn$overall["Accuracy"]) * 100,
  Precision = c(cm_lr$byClass["Precision"], cm_rf$byClass["Precision"],
                cm_svm$byClass["Precision"], cm_nn$byClass["Precision"]) * 100,
  Recall    = c(cm_lr$byClass["Recall"], cm_rf$byClass["Recall"],
                cm_svm$byClass["Recall"], cm_nn$byClass["Recall"]) * 100,
  F1        = c(cm_lr$byClass["F1"], cm_rf$byClass["F1"],
                cm_svm$byClass["F1"], cm_nn$byClass["F1"]) * 100,
  AUC       = c(as.numeric(auc_lr), as.numeric(auc_rf),
                as.numeric(auc_svm), as.numeric(auc_nn)) * 100
)

metriques_long <- melt(metriques, id.vars = "Model")

ggplot(metriques_long, aes(x = variable, y = value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = paste0(round(value,1),"%")),
            position = position_dodge(width = 0.9), vjust = -0.3, size = 3) +
  scale_fill_manual(values = c("#9C27B0","#4CAF50","#F44336","#2196F3")) +
  labs(title = "Comparison of Performance Metrics - 4 Models",
       subtitle = "Note: Recall ~50-60% — upsampling improves vs. no balancing; SMOTE could push further",
       x = "Metric", y = "Value (%)", fill = "Model") +
  theme_minimal(base_size = 12) + ylim(0, 120)
sauvegarder_graph("21_comparaison_metriques", largeur = 12)

# ---- GRAPHIQUE 22 : Matrices de confusion (2x2) ----------------------------
plot_cm <- function(cm, titre) {
  df_cm <- as.data.frame(cm$table)
  names(df_cm) <- c("Predicted", "Actual", "Freq")
  ggplot(df_cm, aes(x = Predicted, y = Actual, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), size = 6, fontface = "bold") +
    scale_fill_gradient(low = "#E3F2FD", high = "#1565C0") +
    labs(title = titre, x = "Predicted", y = "Actual") +
    theme_minimal(base_size = 11) + theme(legend.position = "none")
}

grid.arrange(
  plot_cm(cm_lr,  "Logistic Regression"),
  plot_cm(cm_rf,  "Random Forest"),
  plot_cm(cm_svm, "SVM"),
  plot_cm(cm_nn,  "Neural Network"),
  ncol = 2,
  top  = "Confusion Matrices - 4 Models (Test Set)"
)
sauvegarder_graph("22_matrices_confusion", largeur = 10, hauteur = 8)


# =============================================================================
# PARTIE 7 : ANALYSE DE SURVIE (KAPLAN-MEIER + COX)
# =============================================================================
cat("\n========== PARTIE 7 : ANALYSE DE SURVIE ==========\n")

surv_obj <- Surv(time = donnees_clean$tenure, event = donnees_clean$Churn_bin)

# ---- GRAPHIQUE 23 : KM global -----------------------------------------------
km_global <- survfit(surv_obj ~ 1, data = donnees_clean)

p_km_global <- ggsurvplot(
  km_global, data = donnees_clean,
  conf.int = TRUE, risk.table = TRUE,
  palette = "#1565C0", ggtheme = theme_minimal(base_size = 12),
  title = "Global Kaplan-Meier Survival Curve",
  xlab = "Tenure (months)", ylab = "Retention Probability",
  risk.table.title = "Customers at Risk", surv.median.line = "hv"
)
png("graphiques/23_survie_globale.png", width = 1200, height = 900, res = 120)
print(p_km_global)
dev.off()
cat("  -> Graphique sauvegardé : 23_survie_globale\n")

# ---- GRAPHIQUE 24 : KM par contrat ------------------------------------------
km_contrat <- survfit(surv_obj ~ Contract, data = donnees_clean)
p_km_contrat <- ggsurvplot(
  km_contrat, data = donnees_clean, conf.int = TRUE, pval = TRUE,
  risk.table = TRUE, legend.labs = levels(donnees_clean$Contract),
  palette = c("#F44336","#FF9800","#4CAF50"), ggtheme = theme_minimal(base_size = 12),
  title = "Survival by Contract Type",
  xlab = "Tenure (months)", ylab = "Retention Probability"
)
png("graphiques/24_survie_contrat.png", width = 1300, height = 1000, res = 120)
print(p_km_contrat)
dev.off()
cat("  -> Graphique sauvegardé : 24_survie_contrat\n")

# ---- GRAPHIQUE 25 : KM par Internet -----------------------------------------
km_internet <- survfit(surv_obj ~ InternetService, data = donnees_clean)
p_km_internet <- ggsurvplot(
  km_internet, data = donnees_clean, conf.int = TRUE, pval = TRUE,
  risk.table = TRUE, legend.labs = levels(donnees_clean$InternetService),
  palette = c("#F44336","#2196F3","#4CAF50"), ggtheme = theme_minimal(base_size = 12),
  title = "Survival by Internet Service",
  xlab = "Tenure (months)", ylab = "Retention Probability"
)
png("graphiques/25_survie_internet.png", width = 1300, height = 1000, res = 120)
print(p_km_internet)
dev.off()
cat("  -> Graphique sauvegardé : 25_survie_internet\n")

# ---- GRAPHIQUE 26 : KM Senior vs Non-Senior ---------------------------------
km_senior <- survfit(surv_obj ~ SeniorCitizen, data = donnees_clean)
p_km_senior <- ggsurvplot(
  km_senior, data = donnees_clean, conf.int = TRUE, pval = TRUE,
  risk.table = TRUE, legend.labs = c("Non-Senior","Senior"),
  palette = c("#2196F3","#F44336"), ggtheme = theme_minimal(base_size = 12),
  title = "Survival: Senior vs Non-Senior",
  xlab = "Tenure (months)", ylab = "Retention Probability"
)
png("graphiques/26_survie_senior.png", width = 1300, height = 1000, res = 120)
print(p_km_senior)
dev.off()
cat("  -> Graphique sauvegardé : 26_survie_senior\n")

# ---- 7.5 Modèle de Cox ------------------------------------------------------
cat("\n--- Modèle de Cox ---\n")
cox_model <- coxph(
  Surv(tenure, Churn_bin) ~ MonthlyCharges + Contract + InternetService +
    SeniorCitizen + Partner + Dependents + PaperlessBilling,
  data = donnees_clean
)
print(summary(cox_model))

# ---- GRAPHIQUE 27 : Hazard Ratios -------------------------------------------
cox_res <- as.data.frame(summary(cox_model)$coefficients)
cox_ci  <- as.data.frame(summary(cox_model)$conf.int)
cox_res$Variable    <- rownames(cox_res)
cox_res$HR          <- cox_res[, 2]
cox_res$lower       <- cox_ci[, 3]
cox_res$upper       <- cox_ci[, 4]
cox_res$significant <- cox_res[, 5] < 0.05

# Note : l'HR de MonthlyCharges est < 1 (HR=0.971) alors que les churners ont
# en moyenne des charges plus élevées en analyse descriptive.
# Ce renversement d'effet est typique : après contrôle du type de contrat
# et du service internet (qui sont confondeurs : fibre = prix élevé + fort churn),
# l'effet net de MonthlyCharges seul devient légèrement protecteur.
# Il s'agit d'une association conditionnelle, NON d'un effet causal.

ggplot(cox_res, aes(x = reorder(Variable, HR), y = HR, color = significant)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.3, linewidth = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey50") +
  coord_flip() +
  scale_color_manual(values = c("FALSE" = "grey60", "TRUE" = "#D32F2F"),
                     labels = c("Not significant", "Significant (p<0.05)")) +
  labs(
    title    = "Cox Model - Hazard Ratios (HR)",
    subtitle = "HR > 1: associated with higher churn risk  |  HR < 1: associated with lower risk",
    x = "", y = "Hazard Ratio", color = ""
  ) +
  theme_minimal(base_size = 12) + theme(legend.position = "bottom")
sauvegarder_graph("27_cox_hazard_ratios", largeur = 11, hauteur = 8)

# ---- 7.6 Test de l'hypothèse des risques proportionnels (Schoenfeld) --------
# Le modèle de Cox suppose que les HR sont constants dans le temps.
# Le test de Schoenfeld vérifie cette hypothèse : p > 0.05 -> hypothèse acceptable.
cat("\n--- Test de Schoenfeld (hypothèse de proportionnalité) ---\n")
cox_zph <- cox.zph(cox_model)
print(cox_zph)

# ---- GRAPHIQUE 28 : Résidus de Schoenfeld -----------------------------------
p_schoenfeld <- ggcoxzph(cox_zph, font.main = 10, ggtheme = theme_minimal(base_size = 9))

png("graphiques/28_schoenfeld_test.png", width = 1800, height = 1200, res = 120)
do.call(gridExtra::grid.arrange, c(p_schoenfeld, ncol = 3,
  list(top = "Schoenfeld Residuals - Proportional Hazards Assumption Check\n(Flat line = PH assumption satisfied)")))
dev.off()
cat("  -> Graphique sauvegardé : 28_schoenfeld_test\n")


# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
cat("\n")
cat("=============================================================\n")
cat("  ANALYSE TERMINÉE\n")
cat("=============================================================\n")
cat(sprintf("  Clients analysés             : %d\n", nrow(donnees_clean)))
cat(sprintf("  Taux de churn                : %.1f%%\n", taux_churn))
cat(sprintf("  AUC Logistic Regression      : %.4f (baseline)\n", as.numeric(auc_lr)))
cat(sprintf("  AUC Random Forest            : %.4f\n", as.numeric(auc_rf)))
cat(sprintf("  AUC SVM                      : %.4f\n", as.numeric(auc_svm)))
cat(sprintf("  AUC Neural Network           : %.4f\n", as.numeric(auc_nn)))
cat(sprintf("  Recall Random Forest         : %.1f%%\n",
            cm_rf$byClass["Recall"] * 100))
cat("  Nombre de graphiques         : 28\n")
cat("  Dossier de sortie            : graphiques/\n")
cat("=============================================================\n")

fichiers <- list.files("graphiques/", pattern = "\\.png$")
for (f in sort(fichiers)) cat(" -", f, "\n")
