library(keras)
library(tfdatasets)
library(tensorflow)
library(purrr)
install_keras(tensorflow = "gpu")

# Loading the MNIST dataset in Keras

mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y

train_images <- array_reshape(train_images, c(60000, 28, 28, 1))

# Convert the integers to floats
train_images <- train_images %>% k_cast(dtype = "float32")

# normalize images to [-1, 1] because the generator uses tanh activation
train_images <- (train_images - 127.5) / 127.5

buffer_size <- 60000
batch_size <- 256
batches_per_epoch <- (buffer_size / batch_size) %>% round()

train_dataset <- tensor_slices_dataset(train_images) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_batch(batch_size)

make_generator_model <- function() {
  
  model <- keras_model_sequential()  %>%
    
    layer_dense(units = 7 * 7 * 64, use_bias = FALSE) %>%
    
    layer_batch_normalization() %>%
    
    layer_activation_leaky_relu() %>%
    
    layer_reshape(target_shape = c(7, 7, 64)) %>%
    
    layer_conv_2d_transpose(
      filters = 64,
      kernel_size = c(5, 5),
      strides = c(1, 1),
      padding = "same",
      use_bias = FALSE
    ) %>%
    
    layer_batch_normalization() %>%
    
    layer_activation_leaky_relu() %>%
    
    layer_conv_2d_transpose(
      filters = 32,
      kernel_size = c(5, 5),
      strides = c(2, 2),
      padding = "same",
      use_bias = FALSE
    ) %>%
    
    layer_batch_normalization() %>%
    
    layer_activation_leaky_relu() %>%
    
    layer_conv_2d_transpose(
      filters = 1,
      kernel_size = c(5, 5),
      strides = c(2, 2),
      padding = "same",
      use_bias = FALSE,
      activation = "tanh"
    ) 
  
  return(model)
}


make_discriminator_model <- function() {
  
  model <- keras_model_sequential()  %>%
    layer_conv_2d(
      filters = 64,
      kernel_size = c(5, 5),
      strides = c(2, 2),
      padding = "same"
    ) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(rate = 0.3) %>%
    layer_conv_2d(
      filters = 128,
      kernel_size = c(5, 5),
      strides = c(2, 2),
      padding = "same"
    ) %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(rate = 0.3) %>%
    layer_flatten() %>%
    layer_dense(units = 1)
  
  return(model)
}


generator <- make_generator_model()
discriminator <- make_discriminator_model()


discriminator_loss <- function(real_output, generated_output) {
  real_loss <- loss_binary_crossentropy(
    y_true = k_ones_like(real_output),
    y_pred = real_output,
    from_logits = TRUE)
  generated_loss <- loss_binary_crossentropy(
    y_true = k_zeros_like(generated_output),
    y_pred = generated_output,
    from_logits = TRUE)
  real_loss + generated_loss
}


generator_loss <- function(generated_output) {
  loss_binary_crossentropy(
    y_true = k_ones_like(generated_output),
    y_pred = generated_output,
    from_logits = TRUE)
}

discriminator_optimizer <- tf$keras$optimizers$Adam(1e-4)
generator_optimizer <- tf$keras$optimizers$Adam(1e-4)

train <- function(dataset, epochs, noise_dim) {
  for (epoch in seq_len(num_epochs)) {
    start <- Sys.time()
    total_loss_gen <- 0
    total_loss_disc <- 0
    iter <- make_iterator_one_shot(train_dataset)
    
    for(batch_id in 1:batches_per_epoch){
      batch <- iterator_get_next(iter)
      noise <- k_random_normal(c(batch_size, noise_dim))
      
      with(tf$GradientTape() %as% gen_tape, { with(tf$GradientTape() %as% disc_tape, {
        generated_images <- generator(noise, training = TRUE)
        disc_real_output <- discriminator(batch, training = TRUE)
        disc_generated_output <-
          discriminator(generated_images, training = TRUE)
        gen_loss <- generator_loss(disc_generated_output)
        disc_loss <-
          discriminator_loss(disc_real_output, disc_generated_output)
      }) })
      
      gradients_of_generator <-
        gen_tape$gradient(gen_loss, generator$trainable_variables)
      gradients_of_discriminator <-
        disc_tape$gradient(disc_loss, discriminator$trainable_variables)
      
      generator_optimizer$apply_gradients(purrr::transpose(
        list(gradients_of_generator, generator$trainable_variables)
      ))
      discriminator_optimizer$apply_gradients(purrr::transpose(
        list(gradients_of_discriminator, discriminator$trainable_variables)
      ))
      
      total_loss_gen <- total_loss_gen + gen_loss
      total_loss_disc <- total_loss_disc + disc_loss
      
    }
    
    print(paste("Time for epoch ", epoch, ": ", Sys.time() - start))
    print(paste("Generator loss: ", sum(total_loss_gen$numpy()) / (batches_per_epoch * batch_size)))
    print(paste("Discriminator loss: ", sum(total_loss_disc$numpy()) / (batches_per_epoch * batch_size))) 
    
    if(epoch %% 10 == 0){
      
      test_input <- k_random_normal(c(25, noise_dim))
      
      predictions <- generator(test_input, training = FALSE)
      par(mfcol = c(5, 5))
      par(mar = c(0.5, 0.5, 0.5, 0.5),
          xaxs = 'i',
          yaxs = 'i')
      for (i in 1:25) {
        img <- predictions[i, , , 1]
        img <- t(apply(img, 2, rev))
        image(
          1:28,
          1:28,
          img * 127.5 + 127.5,
          col = gray((0:255) / 255),
          xaxt = 'n',
          yaxt = 'n'
        )
      }
      
      save_model_hdf5(generator, file="GAN_MNIST_generator.h5")
    }
  }
}

noise_dim <- 100
num_epochs <- 100
train(train_dataset, num_epochs, noise_dim)

