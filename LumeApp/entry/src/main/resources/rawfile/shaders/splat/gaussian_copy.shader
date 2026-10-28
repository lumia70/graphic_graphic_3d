{
    "compatibility_info": {
        "version": "22.00",
        "type": "shader"
    },
    "category": "3DGS/Copy",
    "displayName": "Gaussian Splat Copy Pass",
    "vert": "splatshaders://gaussian_copy.vert.spv",
    "frag": "splatshaders://gaussian_copy.frag.spv",
    "pipelineLayout": "splatshaders://gaussian_copy.shaderpl",
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
                    "enableBlend": false,
                    "colorWriteMask": "r_bit|g_bit|b_bit|a_bit"
                }
            ]
        }
    }
}
