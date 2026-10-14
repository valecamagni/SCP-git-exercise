
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


N_EVENTS = 50000
PT_MIN = 20.0
ETA_MAX = 2.4
OUTPUT_FILE = "ditau_masses.csv"

# FOUR-VECTOR FUNCTIONS

def pt_eta_phi_to_p4(pt, eta, phi):
    """
    Convert (pt, eta, phi) to (E, px, py, pz).
    Assume massless visible candidates.
    """
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = pt * np.cosh(eta)

    return energy, px, py, pz


def invariant_mass(p4_1, p4_2):
    """Compute invariant mass of two candidates."""

    E = p4_1[0] + p4_2[0]
    px = p4_1[1] + p4_2[1]
    py = p4_1[2] + p4_2[2]
    pz = p4_1[3] + p4_2[3]

    mass_squared = E**2 - px**2 - py**2 - pz**2

    return np.sqrt(np.maximum(mass_squared, 0.0))


# EVENT GENERATION

def generate_events(n_events=10000, seed=42):
    """Generate toy events with two tau candidates."""

    rng = np.random.default_rng(seed)

    events = pd.DataFrame({
        "pt1": rng.uniform(5, 100, n_events),
        "eta1": rng.uniform(-3, 3, n_events),
        "phi1": rng.uniform(-np.pi, np.pi, n_events),

        "pt2": rng.uniform(5, 100, n_events),
        "eta2": rng.uniform(-3, 3, n_events),
        "phi2": rng.uniform(-np.pi, np.pi, n_events),
    })

    return events



# EVENT SELECTION

def select_events(df, pt_min=20, eta_max=2.4):
    """Select events with two accepted tau candidates."""

    mask = (
        (df["pt1"] > pt_min)
        & (df["pt2"] > pt_min)
        & (np.abs(df["eta1"]) < eta_max)
        & (np.abs(df["eta2"]) < eta_max)
    )

    return df.loc[mask].copy()


# MASS RECONSTRUCTION

def reconstruct_masses(df):
    """Reconstruct visible di-tau masses."""

    p4_1 = pt_eta_phi_to_p4(
        df["pt1"].to_numpy(),
        df["eta1"].to_numpy(),
        df["phi1"].to_numpy()
    )

    p4_2 = pt_eta_phi_to_p4(
        df["pt2"].to_numpy(),
        df["eta2"].to_numpy(),
        df["phi2"].to_numpy()
    )

    df = df.copy()
    df["mass"] = invariant_mass(p4_1, p4_2)

    return df


# PLOTTING

def plot_mass_distribution(df):
    """Plot reconstructed di-tau mass distribution."""

    plt.figure(figsize=(8, 6))

    plt.hist(
        df["mass"],
        bins=np.linspace(0, 500, 51),
        histtype="step",
        linewidth=2,
        color="navy",
        label="Toy di-tau events"
    )

    plt.xlabel(r"$m_{\tau\tau}^{vis}$ [GeV]")
    plt.ylabel("Events / 10 GeV")
    plt.title("Reconstructed di-tau mass")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("ditau_mass.png", dpi=150)
    plt.close()



# MAIN

def main():

    print("Generating events...")
    df = generate_events(N_EVENTS)

    n_generated = len(df)

    print("Applying event selection...")
    df = select_events(df, PT_MIN, ETA_MAX)

    n_selected = len(df)

    efficiency = n_selected / n_generated

    print(f"Generated events: {n_generated}")
    print(f"Selected events:  {n_selected}")
    print(f"Efficiency:       {efficiency:.3f}")

    print("Reconstructing masses...")
    df = reconstruct_masses(df)

    print(f"Mean mass: {df['mass'].mean():.2f} GeV")

    df.to_csv(OUTPUT_FILE, index=False)

    plot_mass_distribution(df)

    print("Results saved successfully!")


if __name__ == "__main__":
    main()
