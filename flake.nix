{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShell = pkgs.mkShellNoCC {
          buildInputs = [
            pkgs.clang-tools # Must come before `pkgs.clang` for clangd to work.

            pkgs.clang
            pkgs.nixfmt-rfc-style
            pkgs.paps
            pkgs.poppler-utils
            pkgs.tree
          ];
        };
      }
    );
}
