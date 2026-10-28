/*
 * Minimal securec.h shim for OHOS NDK builds.
 *
 * Provides memcpy_s, memmove_s, memset_s (used by LumeBase allocator.h)
 * and vsnprintf_s (used by LumeEngine logger.cpp).
 *
 * The callers already validate arguments before calling these functions,
 * so the shim simply delegates to standard C library calls after a
 * basic size check and returns 0 (success) or non-zero (failure).
 */
#ifndef SECUREC_H_SHIM
#define SECUREC_H_SHIM

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cerrno>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef EOK
#define EOK 0
#endif

static inline int memcpy_s(void *dest, size_t destMax, const void *src, size_t count)
{
    if (!dest || !src || count > destMax) {
        return EINVAL;
    }
    memcpy(dest, src, count);
    return EOK;
}

static inline int memmove_s(void *dest, size_t destMax, const void *src, size_t count)
{
    if (!dest || !src || count > destMax) {
        return EINVAL;
    }
    memmove(dest, src, count);
    return EOK;
}

static inline int memset_s(void *dest, size_t destMax, int ch, size_t count)
{
    if (!dest || count > destMax) {
        return EINVAL;
    }
    memset(dest, ch, count);
    return EOK;
}

static inline int vsnprintf_s(char *strDest, size_t destMax, size_t count, const char *format, va_list ap)
{
    if (!strDest || destMax == 0) {
        return -1;
    }
    size_t maxLen = (count < destMax) ? count : (destMax - 1);
    int ret = vsnprintf(strDest, maxLen + 1, format, ap);
    if (ret < 0) {
        strDest[0] = '\0';
    }
    return ret;
}

static inline int snprintf_s(char *strDest, size_t destMax, size_t count, const char *format, ...)
{
    if (!strDest || destMax == 0) {
        return -1;
    }
    va_list ap;
    va_start(ap, format);
    int ret = vsnprintf_s(strDest, destMax, count, format, ap);
    va_end(ap);
    return ret;
}

static inline int sprintf_s(char *strDest, size_t destMax, const char *format, ...)
{
    if (!strDest || destMax == 0) {
        return -1;
    }
    va_list ap;
    va_start(ap, format);
    int ret = vsnprintf(strDest, destMax, format, ap);
    va_end(ap);
    if (ret < 0) {
        strDest[0] = '\0';
    }
    return ret;
}

static inline int strcpy_s(char *strDest, size_t destMax, const char *strSrc)
{
    if (!strDest || !strSrc || destMax == 0) {
        return EINVAL;
    }
    size_t srcLen = strlen(strSrc);
    if (srcLen >= destMax) {
        strDest[0] = '\0';
        return ERANGE;
    }
    memcpy(strDest, strSrc, srcLen + 1);
    return EOK;
}

static inline int strncpy_s(char *strDest, size_t destMax, const char *strSrc, size_t count)
{
    if (!strDest || destMax == 0) {
        return EINVAL;
    }
    if (!strSrc) {
        strDest[0] = '\0';
        return EINVAL;
    }
    size_t srcLen = strlen(strSrc);
    size_t copyLen = (count < srcLen) ? count : srcLen;
    if (copyLen >= destMax) {
        strDest[0] = '\0';
        return ERANGE;
    }
    memcpy(strDest, strSrc, copyLen);
    strDest[copyLen] = '\0';
    return EOK;
}

static inline int strcat_s(char *strDest, size_t destMax, const char *strSrc)
{
    if (!strDest || !strSrc || destMax == 0) {
        return EINVAL;
    }
    size_t destLen = strlen(strDest);
    size_t srcLen = strlen(strSrc);
    if (destLen + srcLen >= destMax) {
        return ERANGE;
    }
    memcpy(strDest + destLen, strSrc, srcLen + 1);
    return EOK;
}

#ifdef __cplusplus
}
#endif

#endif /* SECUREC_H_SHIM */
