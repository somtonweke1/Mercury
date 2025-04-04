import numpy as np
import pandas as pd
from typing import Optional, Tuple
import tensorflow as tf
from sklearn.impute import KNNImputer
from .config import Config
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Hybrid imputation processor combining k-NN and GAN approaches."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.knn = KNNImputer(n_neighbors=5)
        self.gan = self._build_gan()
        
    def _build_gan(self) -> tf.keras.Model:
        """Build Wasserstein GAN for structural gap filling."""
        generator = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, 1)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh')
        ])
        
        discriminator = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, 1)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        return GANModel(generator, discriminator)
    
    def process(self, df: pd.DataFrame, target_col: str = 'price') -> pd.DataFrame:
        """
        Apply hybrid imputation to handle missing values.
        
        Args:
            df: Input DataFrame with missing values
            target_col: Column to impute
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        # 1. Identify gap types
        gap_mask = df[target_col].isna()
        structural_mask = self._identify_structural_gaps(df[target_col])
        local_mask = gap_mask & ~structural_mask
        
        # 2. Handle local gaps with k-NN
        if local_mask.any():
            logger.info(f"Filling {local_mask.sum()} local gaps with k-NN")
            local_values = self._apply_knn(df[target_col], local_mask)
            df.loc[local_mask, target_col] = local_values
        
        # 3. Handle structural gaps with GAN
        if structural_mask.any():
            logger.info(f"Filling {structural_mask.sum()} structural gaps with GAN")
            structural_values = self._apply_gan(df[target_col], structural_mask)
            df.loc[structural_mask, target_col] = structural_values
        
        # 4. Add quality metrics
        df['imputation_type'] = np.where(
            structural_mask, 'gan',
            np.where(local_mask, 'knn', 'original')
        )
        
        return df
    
    def _identify_structural_gaps(self, series: pd.Series, threshold: int = 5) -> pd.Series:
        """Identify structural gaps (≥threshold consecutive NaNs)."""
        return series.isna().rolling(window=threshold).sum() >= threshold
    
    def _apply_knn(self, series: pd.Series, mask: pd.Series) -> np.ndarray:
        """Apply k-NN imputation to local gaps."""
        data = series.values.reshape(-1, 1)
        imputed = self.knn.fit_transform(data)
        return imputed[mask]
    
    def _apply_gan(self, series: pd.Series, mask: pd.Series) -> np.ndarray:
        """Apply GAN-based synthesis to structural gaps."""
        # Prepare sequences for GAN
        sequences = self._prepare_sequences(series, mask)
        
        # Generate synthetic data
        synthetic = self.gan.generator.predict(sequences)
        
        return synthetic.flatten()
    
    def _prepare_sequences(self, series: pd.Series, mask: pd.Series,
                         seq_length: int = 10) -> np.ndarray:
        """Prepare sequences for GAN input."""
        values = series.values
        sequences = []
        
        for i in range(len(values) - seq_length + 1):
            if mask[i:i+seq_length].any():
                seq = values[i:i+seq_length]
                sequences.append(seq)
                
        return np.array(sequences).reshape(-1, seq_length, 1)


class GANModel(tf.keras.Model):
    """Wasserstein GAN implementation for time series."""
    
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.disc_optimizer = tf.keras.optimizers.Adam(1e-4)
        
    def compile(self):
        super().compile()
        self.gen_loss_tracker = tf.keras.metrics.Mean(name='gen_loss')
        self.disc_loss_tracker = tf.keras.metrics.Mean(name='disc_loss')
    
    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]
    
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, tf.shape(real_data)[1], 1])
        
        # Train discriminator
        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(fake_data, training=True)
            disc_loss = self._discriminator_loss(real_output, fake_output)
            
        grads = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables)
        )
        
        # Train generator
        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            fake_output = self.discriminator(fake_data, training=True)
            gen_loss = self._generator_loss(fake_output)
            
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )
        
        self.gen_loss_tracker.update_state(gen_loss)
        self.disc_loss_tracker.update_state(disc_loss)
        
        return {
            "gen_loss": self.gen_loss_tracker.result(),
            "disc_loss": self.disc_loss_tracker.result()
        }
    
    def _discriminator_loss(self, real_output, fake_output):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
    
    def _generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output) 