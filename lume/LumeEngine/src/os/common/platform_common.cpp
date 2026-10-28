/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <core/log.h>
#include <core/namespace.h>
#include <core/os/intf_platform.h>
#include <core/os/platform_create_info.h>

#include <base/containers/string.h>
#include <base/containers/string_view.h>

#include "os/platform.h"

CORE_BEGIN_NAMESPACE()

struct PlatformData {
    /** Core root path */
    BASE_NS::string coreRootPath = "./";
    /** Core plugin path (defaults to coreRootPath if not provided separately) */
    BASE_NS::string corePluginPath = "./";
    /** Application root path */
    BASE_NS::string appRootPath = "./";
    /** Application plugin path */
    BASE_NS::string appPluginPath = "./";
};

/** IPlatform implementation for the common (non-NAPI) variant. */
class PlatformCommon final : public IPlatform {
public:
    explicit PlatformCommon(PlatformCreateInfo const& createInfo);
    ~PlatformCommon() override;

    const PlatformData& GetPlatformData() const override;
    void RegisterPluginLocations(IPluginRegister& registry) override;

private:
    PlatformData plat_;
};

PlatformCommon::PlatformCommon(PlatformCreateInfo const& createInfo)
{
    plat_.coreRootPath = createInfo.coreRootPath;
    plat_.corePluginPath = createInfo.coreRootPath;
    plat_.appRootPath = createInfo.appRootPath;
    plat_.appPluginPath = createInfo.appPluginPath;
}

PlatformCommon::~PlatformCommon() {}

const PlatformData& PlatformCommon::GetPlatformData() const
{
    return plat_;
}

void PlatformCommon::RegisterPluginLocations(IPluginRegister& registry)
{
    constexpr BASE_NS::string_view fileproto("file://");
    if (!plat_.corePluginPath.empty()) {
        registry.RegisterPluginPath(fileproto + plat_.corePluginPath);
    }
    if (!plat_.appPluginPath.empty()) {
        if (plat_.appPluginPath != plat_.corePluginPath) {
            registry.RegisterPluginPath(fileproto + plat_.appPluginPath);
        }
    }
}

CORE_NS::IPlatform::Ptr Platform::Create(PlatformCreateInfo const& createInfo)
{
    return CORE_NS::IPlatform::Ptr(new PlatformCommon(createInfo));
}

const CORE_NS::PlatformCreateExtensionInfo* Platform::Extensions(PlatformCreateInfo const& createInfo)
{
    return createInfo.extensions;
}

CORE_END_NAMESPACE()
