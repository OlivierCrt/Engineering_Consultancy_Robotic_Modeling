# Affichage de 3 courbes
#INPUT:
#     f1 f2, f3 les 3 courbes à afficher
#     t liste des temps de communation pour lesquels on veut afficher un segment vertical (sauf en t=0)
#     tim liste des temps de calcul des fcts
#     refi affichage du nom de la fonction sur la sous-figure
######################################################################
def affichage3courbes(t, tim, f1, ref1, f2, ref2, f3, ref3):
    fig, axes = plt.subplots(nrows=3,ncols=1)
    axes[0].plot(tim, f1, "r-")
    axes[0].axhline(y=0)
    axes[0].axvline(x=t[1])
    axes[0].axvline(x=t[2])
    axes[0].axvline(x=t[3])
    axes[0].axvline(x=t[4])
    axes[0].axvline(x=t[5])
    axes[0].set_xlim([0, t[-1]])
    axes[0].set_title('Fonction' + ref1)
    axes[1].plot(tim, f2, "b-")
    axes[1].axhline(y=0)
    axes[1].axvline(x=t[1])
    axes[1].axvline(x=t[2])
    axes[1].axvline(x=t[3])
    axes[1].axvline(x=t[4])
    axes[1].axvline(x=t[5])
    axes[1].set_xlim([0, t[-1]])
    axes[1].set_title('Fonction' + ref2)
    axes[2].plot(tim, f3, "g-")
    axes[2].axhline(y=0)
    axes[2].axvline(x=t[1])
    axes[2].axvline(x=t[2])
    axes[2].axvline(x=t[3])
    axes[2].axvline(x=t[4])
    axes[2].axvline(x=t[5])
    axes[2].set_xlim([0, t[-1]])
    axes[2].set_title('Fonction' + ref3)
    plt.tight_layout()
    plt.show()
    return
