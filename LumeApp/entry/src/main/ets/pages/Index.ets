import lumeapp from 'liblumeapp.so';

@Entry
@Component
struct Index {
  private xComponentId: string = 'lumeapp_xcomponent';
  private intervalId: number = -1;

  aboutToAppear(): void {
    let context = getContext(this) as Context;
    let resMgr = context.resourceManager;
    let filesDir = context.filesDir;
    lumeapp.initResourceManager(resMgr, filesDir);
    console.info('[LumeApp] ResourceManager + filesDir passed to native: ' + filesDir);
  }

  build() {
    Stack() {
      XComponent({
        id: this.xComponentId,
        type: XComponentType.SURFACE,
        libraryname: 'lumeapp'
      })
        .onLoad(() => {
          console.info('[LumeApp] XComponent onLoad — starting render loop');
          this.intervalId = setInterval(() => {
            lumeapp.drawFrame();
          }, 16);
        })
        .onDestroy(() => {
          console.info('[LumeApp] XComponent onDestroy');
          if (this.intervalId >= 0) {
            clearInterval(this.intervalId);
            this.intervalId = -1;
          }
        })
        .width('100%')
        .height('100%')
    }
    .width('100%')
    .height('100%')
  }
}
