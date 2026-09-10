suppressPackageStartupMessages({
  library(dplyr)
  library(future)
  library(furrr)
  library(ggplot2)
  library(ggrepel)
  library(gridExtra)
  library(htmlwidgets)
  library(igraph)
  library(MCMCpack)
  library(nbamen)
  library(plotly)
  library(purrr)
})

source("Code/R/utils.R")


# Data --------------------------------------------------------------------

scores <- read.csv("Data/scores.csv")
excluded_names <- scores$s_l_name[
  scores$csts == 0 | scores$test_total2 == 0
]
scores_AL_d <- scores[!unique(scores$s_l_name) %in% excluded_names, ]
delete_nm <- excluded_names

read_network <- function(path) {
  data <- read.csv(path, check.names = FALSE)
  network <- as.matrix(data[, -1])
  rownames(network) <- colnames(network)
  storage.mode(network) <- "double"
  diag(network) <- 0
  network
}

network_exp <- read_network("Data/Network_EXP.csv")
network_eoi <- read_network("Data/Network_EOI.csv")

stopifnot(
  identical(dim(network_exp), c(20L, 20L)),
  identical(dim(network_eoi), c(20L, 20L)),
  sum(network_exp) == 456,
  sum(network_eoi) == 981,
  sum(network_exp > 0) == 55,
  sum(network_eoi > 0) == 55
)

table_dir <- "Results/Main/Tables"
figure_dir <- "Results/Main/Figures"
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)


# Centrality --------------------------------------------------------------

compute_centrality <- function(network) {
  graph <- graph_from_adjacency_matrix(
    network,
    mode = "directed",
    weighted = TRUE
  )
  pagerank <- page_rank(graph)$vector
  hits <- hits_scores(graph, weights = E(graph)$weight)

  data.frame(
    actor = names(pagerank),
    pagerank = pagerank,
    in_strength = strength(graph, mode = "in"),
    out_strength = strength(graph, mode = "out"),
    betweenness = betweenness(
      graph,
      directed = TRUE,
      weights = 1 / E(graph)$weight
    ),
    closeness = closeness(
      graph,
      mode = "all",
      weights = 1 / E(graph)$weight,
      normalized = FALSE
    ),
    eigen = eigen_centrality(
      graph,
      directed = TRUE,
      weights = E(graph)$weight
    )$vector,
    hub = hits$hub,
    authority = hits$authority,
    row.names = NULL,
    check.names = FALSE
  )
}

centrality_exp <- compute_centrality(network_exp)
centrality_eoi <- compute_centrality(network_eoi)

stopifnot(
  centrality_exp$actor[which.max(centrality_exp$pagerank)] == "s17",
  centrality_exp$actor[which.max(centrality_exp$in_strength)] == "s17",
  centrality_exp$actor[which.max(centrality_exp$out_strength)] == "s17",
  centrality_eoi$actor[which.max(centrality_eoi$pagerank)] == "s10",
  centrality_eoi$actor[which.max(centrality_eoi$in_strength)] == "s06",
  centrality_eoi$actor[which.max(centrality_eoi$out_strength)] == "s17"
)

write.csv(
  centrality_exp,
  file.path(table_dir, "EXP_centrality.csv"),
  row.names = FALSE
)
write.csv(
  centrality_eoi,
  file.path(table_dir, "EOI_centrality.csv"),
  row.names = FALSE
)

draw_network(
  network_exp,
  option = "pre",
  path = file.path(figure_dir, "graph_exp.png"),
  seed = 204
)
draw_network(
  network_exp,
  option = "post",
  path = file.path(figure_dir, "graph_exp_p.png"),
  seed = 204
)
draw_network(
  network_eoi,
  option = "pre",
  path = file.path(figure_dir, "graph_eoi.png"),
  seed = 685
)
draw_network(
  network_eoi,
  option = "post",
  path = file.path(figure_dir, "graph_eoi_p.png"),
  seed = 685
)

N <- nrow(network_exp)
X <- array(0, c(N, N, 1))


# AMEN model ---------------------------------------------------------------

num_runs <- 10
num_workers <- 8
niter <- 100000
nburn <- 10000
nthin <- 5

run_amen <- function(data, dimension, prior_sd_r) {
  amen_count_nb(
    data = data,
    X = X,
    niter = niter,
    nburn = nburn,
    nthin = nthin,
    overdispersion = TRUE,
    zeroinflate = FALSE,
    jump_r = 0.5,
    r_init = 1,
    verbose = FALSE,
    direct = TRUE,
    fix = TRUE,
    fixsd = TRUE,
    jump_gamma = 0.5,
    jump_beta = 2,
    jump_alpha = 2,
    jump_z = 0.6,
    jump_w = 0.6,
    pr_sd_z = 1,
    pr_sd_w = 1,
    pr_sd_beta = 3,
    pr_sd_alpha = 3,
    pr_mean_r = log(1),
    pr_sd_r = prior_sd_r,
    eta_init = 0.5,
    pr_a_eta = 1,
    pr_b_eta = 1,
    jump_eta = 0.3,
    pr_mean_gamma = 0.5,
    pr_sd_gamma = 1,
    ndim = dimension,
    singledist = FALSE,
    single = FALSE,
    hierarchical_r = FALSE,
    vector = TRUE,
    missing = -99,
    covariate = FALSE,
    fix_r = FALSE
  )
}

fit_chains <- function(network, dimension, prior_sd_r) {
  options(parallelly.maxWorkers.localhost = num_workers)
  if (.Platform$OS.type == "windows") {
    plan(multisession, workers = num_workers)
  } else {
    plan(multicore, workers = num_workers)
  }
  on.exit(plan(sequential), add = TRUE)

  set.seed(1234)
  fits <- future_map(
    seq_len(num_runs),
    ~ run_amen(network, dimension, prior_sd_r),
    .options = furrr_options(seed = TRUE)
  )

  reference_chain <- which.min(vapply(fits, function(fit) fit$bic, numeric(1)))
  aligned_fits <- fits

  for (index in setdiff(seq_len(num_runs), reference_chain)) {
    aligned <- procrustes_mat(
      aligned_fits[[index]], aligned_fits[[reference_chain]]
    )
    aligned_fits[[index]]$z_estimate <- aligned$z_estimate
    aligned_fits[[index]]$w_estimate <- aligned$w_estimate
    aligned_fits[[index]]$z <- aligned$z
    aligned_fits[[index]]$w <- aligned$w
  }

  stopifnot(all(vapply(aligned_fits, function(fit) {
    identical(unique(as.numeric(fit$gamma)), 1)
  }, logical(1))))

  list(fits = aligned_fits, reference_chain = reference_chain)
}

# Dimensions and dispersion-prior SDs are the minimum-mean-BIC choices.
model_exp <- fit_chains(network_exp, dimension = 2, prior_sd_r = 0.10)
model_eoi <- fit_chains(network_eoi, dimension = 3, prior_sd_r = 0.10)


# Latent-position figures -------------------------------------------------

reference_exp <- model_exp$fits[[model_exp$reference_chain]]
png(
  file.path(figure_dir, "latent_exp.png"),
  width = 3600,
  height = 1800,
  res = 300
)
plot_model(
  reference_exp,
  xrange = c(-2.5, 2.5),
  yrange = c(-2.5, 2.5),
  fix = FALSE,
  b = 0
)
dev.off()

reference_eoi <- model_eoi$fits[[model_eoi$reference_chain]]
eoi_sender <- plot_model(
  reference_eoi,
  xrange = c(-2.5, 2.5),
  yrange = c(-2.5, 2.5),
  fix = FALSE,
  b = 0,
  pos = "z"
)
eoi_receiver <- plot_model(
  reference_eoi,
  xrange = c(-2.5, 2.5),
  yrange = c(-2.5, 2.5),
  fix = FALSE,
  b = 0,
  pos = "w"
)

eoi_camera <- list(
  eye = list(x = 0, y = 0, z = 1.45),
  up = list(x = 0, y = 1, z = 0),
  projection = list(type = "perspective")
)
prepare_eoi_widget <- function(figure, title, label_overrides = character()) {
  label_positions <- setNames(
    rep("top right", nrow(reference_eoi$data)),
    rownames(reference_eoi$data)
  )
  label_positions[names(label_overrides)] <- label_overrides

  figure %>%
    style(
      marker = list(color = "blue", size = 7),
      textfont = list(
        size = 32,
        color = "black",
        family = "Arial Black"
      ),
      textposition = unname(label_positions),
      traces = 1
    ) %>%
    layout(
      title = list(
        text = title,
        font = list(size = 40, color = "black"),
        x = 0.04,
        y = 0.98,
        xanchor = "left"
      ),
      scene = list(
        camera = eoi_camera,
        aspectmode = "manual",
        aspectratio = list(x = 1, y = 1, z = 0.08),
        xaxis = list(visible = FALSE),
        yaxis = list(visible = FALSE),
        zaxis = list(visible = FALSE)
      ),
      paper_bgcolor = "white",
      plot_bgcolor = "white",
      margin = list(l = 0, r = 0, t = 50, b = 0)
    )
}
eoi_sender <- prepare_eoi_widget(
  eoi_sender,
  "Sender position",
  c(s02 = "top left", s08 = "bottom right")
)
eoi_receiver <- prepare_eoi_widget(
  eoi_receiver,
  "Receiver position",
  c(
    s13 = "top left",
    s16 = "bottom right",
    s14 = "bottom left",
    s12 = "top right"
  )
)
saveWidget(
  eoi_sender,
  file.path(figure_dir, "latent_eoi_sender.html"),
  selfcontained = TRUE
)
saveWidget(
  eoi_receiver,
  file.path(figure_dir, "latent_eoi_receiver.html"),
  selfcontained = TRUE
)


# Network mediation -------------------------------------------------------

mediation_exp <- make_multi_chain_med(model_exp$fits, scores = scores_AL_d)
mediation_eoi <- make_multi_chain_med(model_eoi$fits, scores = scores_AL_d)

mediation_results <- bind_rows(
  data.frame(
    network = "EXP", effect = rownames(mediation_exp), mediation_exp,
    row.names = NULL, check.names = FALSE
  ),
  data.frame(
    network = "EOI", effect = rownames(mediation_eoi), mediation_eoi,
    row.names = NULL, check.names = FALSE
  )
)


write.csv(
  mediation_results,
  file.path(table_dir, "amen_mediation.csv"),
  row.names = FALSE
)
