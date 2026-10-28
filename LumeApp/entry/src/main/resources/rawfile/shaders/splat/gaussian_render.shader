{
    "compatibility_info": {
        "version": "22.00",
        "type": "shader"
    },
    "category": "3DGS/Splatting",
    "displayName": "Gaussian Splatting Render",
    "vert": "splatshaders://gaussian_render.vert.spv",
    "frag": "splatshaders://gaussian_render.frag.spv",
    "pipelineLayout": "splatshaders://gaussian_render.shaderpl",
    "vertexInputDeclaration": "",
    "state": {
        "rasterizationState": {
            "enableDepthClamp": false,
            "enableRasterizerDiscard": false,
            "cullModeFlags": "none",
            "frontFace": "counter_clockwise"
        },
        "depthStencilState": {
            "enableDepthTest": false,
            "enableDepthWrite": false
        },
        "colorBlendState": {
            "colorAttachments": [
                {
                    "enableBlend": true,
                    "srcColorBlendFactor": "one",
                    "dstColorBlendFactor": "one_minus_src_alpha",
                    "colorBlendOp": "add",
                    "srcAlphaBlendFactor": "one",
                    "dstAlphaBlendFactor": "one_minus_src_alpha",
                    "alphaBlendOp": "add",
                    "colorWriteMask": "r_bit|g_bit|b_bit|a_bit"
                }
            ]
        }
    }
}
