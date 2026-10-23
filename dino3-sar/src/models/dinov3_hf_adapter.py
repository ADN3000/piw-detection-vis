import torch
import torch.nn as nn


class DINOv3HFAdapter(nn.Module):
    """
    Adaptador para utilizar o DINOv3 carregado pelo
    Hugging Face dentro do INSID3.

    O INSID3 espera que o encoder possua o método:

        get_intermediate_layers()

    A implementação do Hugging Face não fornece esse
    método com a mesma interface do DINOv3 nativo.

    Este adaptador cria essa compatibilidade.
    """

    def __init__(self, model):
        super().__init__()

        # Modelo DINOv3 carregado pelo Hugging Face
        self.model = model

        # ViT-S/16 utiliza patches de 16 x 16 pixels
        self.patch_size = 16

        # DINOv3 ViT-S/16 possui 4 register tokens
        self.num_register_tokens = 4


    def forward(self, x):
        """
        Forward convencional do modelo.
        """

        return self.model(
            pixel_values=x
        )


    def get_intermediate_layers(
        self,
        x,
        n=1,
        reshape=False
    ):
        """
        Implementa a interface esperada pelo INSID3.

        Parâmetros
        ----------
        x:
            Tensor de entrada no formato:

            [B, 3, H, W]

        n:
            Número de últimas camadas que serão retornadas.

        reshape:
            Se True, transforma os patch tokens em um
            mapa espacial:

            [B, C, H_patches, W_patches]

        Retorno
        -------
        tuple contendo os mapas de características.
        """

        # -------------------------------------------------
        # 1. Executar DINOv3 solicitando hidden states
        # -------------------------------------------------

        outputs = self.model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states


        # -------------------------------------------------
        # 2. Selecionar as últimas n camadas
        # -------------------------------------------------

        selected_layers = hidden_states[-n:]

        results = []


        # -------------------------------------------------
        # 3. Processar cada camada selecionada
        # -------------------------------------------------

        for tokens in selected_layers:

            # ---------------------------------------------
            # IMPORTANTE
            #
            # Reproduzimos aqui a normalização final
            # utilizada pelo get_intermediate_layers()
            # da implementação nativa do DINOv3.
            #
            # Formato continua:
            #
            # [B, número_de_tokens, 384]
            # ---------------------------------------------

            tokens = self.model.norm(tokens)


            # ---------------------------------------------
            # 4. Remover tokens que não representam patches
            #
            # Estrutura:
            #
            # CLS
            # ↓
            # [CLS][REG][REG][REG][REG][PATCH][PATCH]...
            #
            # Removemos:
            #
            # 1 CLS token
            # 4 register tokens
            # ---------------------------------------------

            patch_tokens = tokens[
                :,
                1 + self.num_register_tokens:,
                :
            ]


            # ---------------------------------------------
            # 5. Converter para mapa espacial se solicitado
            # ---------------------------------------------

            if reshape:

                B, N, C = patch_tokens.shape

                # Número de patches na vertical
                H = (
                    x.shape[-2]
                    // self.patch_size
                )

                # Número de patches na horizontal
                W = (
                    x.shape[-1]
                    // self.patch_size
                )

                expected_tokens = H * W


                # -----------------------------------------
                # Verificação de segurança
                # -----------------------------------------

                if N != expected_tokens:

                    raise RuntimeError(
                        "\nNúmero inesperado de patch tokens."
                        f"\nRecebido: {N}"
                        f"\nEsperado: {expected_tokens}"
                        f"\nEntrada: "
                        f"{x.shape[-2]} x {x.shape[-1]}"
                        f"\nGrade esperada: {H} x {W}"
                    )


                # -----------------------------------------
                # [B, N, C]
                #
                # N = H * W
                #
                # vira:
                #
                # [B, H, W, C]
                # -----------------------------------------

                patch_tokens = patch_tokens.reshape(
                    B,
                    H,
                    W,
                    C
                )


                # -----------------------------------------
                # INSID3 espera:
                #
                # [B, C, H, W]
                # -----------------------------------------

                patch_tokens = patch_tokens.permute(
                    0,
                    3,
                    1,
                    2
                ).contiguous()


            # ---------------------------------------------
            # 6. Armazenar resultado
            # ---------------------------------------------

            results.append(
                patch_tokens
            )


        # -------------------------------------------------
        # 7. O DINOv3 nativo retorna uma tupla
        # -------------------------------------------------

        return tuple(results)