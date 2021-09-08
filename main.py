import argparse
import re
import matplotlib.pyplot as plt
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich import box
from risiko import RISIKO


def read_input():
    pattern = re.compile("^[0-9]+$")
    while True:
        print("[green]Attacker armies 💣: [/green]")
        a = input()
        print("[green]Defender armies 🛡️: [/green]")
        d = input()
        if pattern.match(a) and pattern.match(d):
            return int(a), int(d)
        else:
            print("[red]These armies cannot fight!\n[/red]")


def plot_heatmap(heatmap):
    plt.imshow(heatmap, cmap="hot", origin="lower")
    plt.xlabel("defender")
    plt.ylabel("attacker")
    plt.colorbar().set_label("win probability")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program that, given the armies size of the attacker and the defender, calculates the probability of winning the battle and the expected losses of the attacker using a Markov stochastic process."
    )
    parser.add_argument(
        "-p",
        "--plot",
        default=False,
        action="store_true",
        help="plot probabilities heatmap",
    )
    args = parser.parse_args()

    console = Console()
    attacker, defender = read_input()
    risiko = RISIKO()
    results = risiko.solve(attacker, defender)

    console.print(
        Panel(
            f"[green]Win probability:[/green] 🔥 [magenta]{results['win_probability'] * 100:.2f}%[/magenta] 🔥",
            box=box.HEAVY,
            expand=False,
            border_style="blue",
        )
    )
    console.print(
        Panel(
            f"[green]Losses expected value:[/green] 💀 [magenta]{results['expected_losses']:.2f}[/magenta] 💀",
            box=box.HEAVY,
            expand=False,
            border_style="blue",
        )
    )
    console.print(
        Panel(
            f"[green]Losses standard deviation:[/green] 💀 [magenta]{results['std_losses']:.2f}[/magenta] 💀",
            box=box.HEAVY,
            expand=False,
            border_style="blue",
        )
    )

    if args.plot:
        plot_heatmap(results["win_probability_matrix"])
