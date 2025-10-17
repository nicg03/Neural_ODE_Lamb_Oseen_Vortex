import torch
import numpy as np
from config import TRAINING_CONFIG, MODEL_CONFIG, PATHS, DATASET_CONFIG, VORTEX_CONFIG

# Funzione esatta del Lamb-Oseen per un singolo vortice
def lamb_oseen_velocity(x, y, gamma, t, nu, x0=0, y0=0):
    # Traslazione delle coordinate
    x = x - x0
    y = y - y0
    r2 = x**2 + y**2
    factor = gamma / (2 * np.pi * r2) * (1 - np.exp(-r2 / (4 * nu * t)))
    u_x = -y * factor
    u_y = x * factor
    return u_x, u_y

# Neural ODE dynamics
class FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 128),  # Aumentato il numero di neuroni
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 2)
        )

    def forward(self, t, xyt):
        x, y, t = xyt[:, 0], xyt[:, 1], xyt[:, 2]
        velocities = self.net(xyt)
        return velocities

def train_model(model, data_in, data_out, config, model_path, metrics_path):
    """
    Train the model and save the results.
    
    Args:
        model: The model to train
        data_in: Input data
        data_out: Target data
        config: Training configuration dictionary
        model_path: Path to save the model
        metrics_path: Path to save the metrics
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'min', 
        patience=config['scheduler_patience'], 
        factor=config['scheduler_factor']
    )

    # Liste per memorizzare le metriche
    train_losses = []
    learning_rates = []

    for epoch in range(config['num_epochs']):
        pred = model(0, data_in)
        loss = loss_fn(pred, data_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        # Salva le metriche
        train_losses.append(loss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        if epoch % config['print_every'] == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

    # Salva le metriche
    np.savez(metrics_path, 
             train_losses=np.array(train_losses),
             learning_rates=np.array(learning_rates))

    # Salva il modello addestrato
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_loss': loss.item()
    }, model_path)

    print("Modello e metriche salvati")
    return train_losses, learning_rates

if __name__ == "__main__":
    # Carica il dataset salvato
    data = torch.load(PATHS['dataset'])
    data_in = data['data_in']
    data_out = data['data_out']

    # Dataset sintetico con tre vortici
    grid = np.linspace(-DATASET_CONFIG['domain_size'], 
                      DATASET_CONFIG['domain_size'], 
                      DATASET_CONFIG['grid_resolution'])
    X, Y = np.meshgrid(grid, grid)

    # Calcolo del campo di velocità totale
    u_x1, u_y1 = lamb_oseen_velocity(X, Y, 
                                    VORTEX_CONFIG['vortex1']['gamma'], 
                                    DATASET_CONFIG['time'], 
                                    DATASET_CONFIG['viscosity'],
                                    VORTEX_CONFIG['vortex1']['x'],
                                    VORTEX_CONFIG['vortex1']['y'])
    
    u_x2, u_y2 = lamb_oseen_velocity(X, Y, 
                                    VORTEX_CONFIG['vortex2']['gamma'], 
                                    DATASET_CONFIG['time'], 
                                    DATASET_CONFIG['viscosity'],
                                    VORTEX_CONFIG['vortex2']['x'],
                                    VORTEX_CONFIG['vortex2']['y'])
    
    u_x3, u_y3 = lamb_oseen_velocity(X, Y, 
                                    VORTEX_CONFIG['vortex3']['gamma'], 
                                    DATASET_CONFIG['time'], 
                                    DATASET_CONFIG['viscosity'],
                                    VORTEX_CONFIG['vortex3']['x'],
                                    VORTEX_CONFIG['vortex3']['y'])

    # Somma dei campi di velocità
    u_x = u_x1 + u_x2 + u_x3
    u_y = u_y1 + u_y2 + u_y3

    # Inizializza e addestra il modello
    model = FeedForward()
    train_model(
        model=model,
        data_in=data_in,
        data_out=data_out,
        config=TRAINING_CONFIG,
        model_path=PATHS['model'],
        metrics_path=PATHS['metrics']
    )
