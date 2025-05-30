# encoding: utf-8


class IPProxyMiddleware(object):
    """
    Proxy IP middleware
    """

    @staticmethod
    def fetch_proxy():
        """
        Get a proxy IP
        """
        # You need to rewrite this function if you want to add proxy pool
        # the function should return an ip in the format of "ip:port" like "12.34.1.4:9090"
        return None

    def process_request(self, request, spider):
        """
        Add the proxy IP to the request
        """
        proxy_data = self.fetch_proxy()
        if proxy_data:
            current_proxy = f'http://{proxy_data}'
            spider.logger.debug(f"current proxy:{current_proxy}")
            request.meta['proxy'] = current_proxy
