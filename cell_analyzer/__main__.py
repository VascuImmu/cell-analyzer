"""python -m cell_analyzer  ->  opens the Cell Analyzer window."""
import multiprocessing


def main():
    from .gui import CellAnalyzerGUI
    CellAnalyzerGUI().mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
