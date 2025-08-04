{ pkgs }: {
    deps = [
        pkgs.python38Full
        pkgs.replitPackages.prybar-python38
        pkgs.replitPackages.stderred
    ];
    env = {
        PYTHON_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            # Needed for pandas / numpy
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            # Needed for pygame
            pkgs.glib
            # Needed for matplotlib
            pkgs.xorg.libX11
        ];
        PYTHONHOME = "${pkgs.python38Full}";
        PYTHONBIN = "${pkgs.python38Full}/bin/python3.8";
        LANG = "en_US.UTF-8";
        STDERREDBIN = "${pkgs.replitPackages.stderred}/bin/stderred";
        PRYBAR_PYTHON_BIN = "${pkgs.replitPackages.prybar-python38}/bin/prybar-python38";
    };
}